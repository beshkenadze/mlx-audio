from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class SileroConfig:
    model_type: str = "silero_vad_v5"
    sample_rate: int = 16000
    chunk_size: int = 512
    context_size: int = 64
    filter_length: int = 256
    hop_length: int = 128
    encoder_channels: List[int] = field(default_factory=lambda: [129, 128, 64, 64, 128])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    encoder_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 1])
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "SileroConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


class SileroVAD(nn.Module):
    def __init__(self, config: SileroConfig):
        super().__init__()
        self.config = config

        self.stft_basis = mx.zeros((config.filter_length + 2, config.filter_length, 1))

        chans = config.encoder_channels
        ks = config.encoder_kernel_sizes
        ss = config.encoder_strides
        self.encoder = []
        for i in range(len(ks)):
            self.encoder.append(
                nn.Conv1d(
                    in_channels=chans[i],
                    out_channels=chans[i + 1],
                    kernel_size=ks[i],
                    stride=ss[i],
                    padding=1,
                )
            )

        self.lstm_Wx = mx.zeros((4 * config.lstm_hidden_size, config.lstm_hidden_size))
        self.lstm_Wh = mx.zeros((4 * config.lstm_hidden_size, config.lstm_hidden_size))
        self.lstm_bias = mx.zeros((4 * config.lstm_hidden_size,))

        self.decoder = nn.Conv1d(
            in_channels=config.lstm_hidden_size,
            out_channels=1,
            kernel_size=1,
            padding=0,
        )

    def _stft(self, x_with_ctx: mx.array) -> mx.array:
        right_pad = self.config.context_size
        x_rev = x_with_ctx[:, -right_pad - 1 : -1][:, ::-1]
        padded = mx.concatenate([x_with_ctx, x_rev], axis=1)

        n_fft = self.config.filter_length
        hop = self.config.hop_length
        n_frames = (padded.shape[1] - n_fft) // hop + 1
        idx = mx.arange(n_fft)[None, :] + (mx.arange(n_frames) * hop)[:, None]
        frames = padded[:, idx]
        B = padded.shape[0]
        frames = frames.reshape(B * n_frames, n_fft, 1)
        out = mx.conv1d(frames, self.stft_basis, stride=1, padding=0)
        out = out.reshape(B, n_frames, n_fft + 2)
        out = out.transpose(0, 2, 1)
        n_freq = n_fft // 2 + 1
        real = out[:, :n_freq, :]
        imag = out[:, n_freq:, :]
        mag = mx.sqrt(real * real + imag * imag + 1e-12)
        return mag

    def _encoder(self, x: mx.array) -> mx.array:
        x = x.transpose(0, 2, 1)
        for layer in self.encoder:
            x = layer(x)
            x = nn.relu(x)
        return x

    def _lstm_step(
        self,
        x_t: mx.array,
        h: mx.array,
        c: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        gates = x_t @ self.lstm_Wx.T + h @ self.lstm_Wh.T + self.lstm_bias
        H = self.config.lstm_hidden_size
        i = mx.sigmoid(gates[:, :H])
        f = mx.sigmoid(gates[:, H:2 * H])
        g = mx.tanh(gates[:, 2 * H:3 * H])
        o = mx.sigmoid(gates[:, 3 * H:])
        c = f * c + i * g
        h = o * mx.tanh(c)
        return h, c

    def __call__(
        self,
        chunk: mx.array,
        h: Optional[mx.array] = None,
        c: Optional[mx.array] = None,
        context: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        B = chunk.shape[0]
        H = self.config.lstm_hidden_size
        ctx_size = self.config.context_size
        if h is None:
            h = mx.zeros((B, H))
        if c is None:
            c = mx.zeros((B, H))
        if context is None:
            context = mx.zeros((B, ctx_size))

        with_ctx = mx.concatenate([context, chunk], axis=1)
        spec = self._stft(with_ctx)
        feats = self._encoder(spec)
        T = feats.shape[1]
        for t in range(T):
            h, c = self._lstm_step(feats[:, t, :], h, c)
        h_3d = h.reshape(B, 1, H)
        out = self.decoder(nn.relu(h_3d))
        prob = mx.sigmoid(out).reshape(B)
        new_context = chunk[:, -ctx_size:]
        return prob, h, c, new_context

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "aufklarer/Silero-VAD-v5-MLX",
        revision: Optional[str] = None,
    ) -> "SileroVAD":
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=repo_id, revision=revision))
        with open(path / "config.json") as f:
            config = SileroConfig.from_dict(json.load(f))
        weights_dict = mx.load(str(path / "model.safetensors"))
        model = cls(config)
        weights = []
        for key, value in weights_dict.items():
            if key == "stft.weight":
                weights.append(("stft_basis", value))
            elif key.startswith("encoder."):
                idx = int(key.split(".")[1])
                if key.endswith(".weight"):
                    weights.append((f"encoder.{idx}.weight", value))
                else:
                    weights.append((f"encoder.{idx}.bias", value))
            elif key == "lstm.Wx":
                weights.append(("lstm_Wx", value))
            elif key == "lstm.Wh":
                weights.append(("lstm_Wh", value))
            elif key == "lstm.bias":
                weights.append(("lstm_bias", value))
            elif key == "decoder.weight":
                weights.append(("decoder.weight", value))
            elif key == "decoder.bias":
                weights.append(("decoder.bias", value))
            else:
                raise KeyError(f"unexpected weight: {key}")
        model.load_weights(weights)
        mx.eval(model.parameters())
        return model


def get_speech_timestamps(
    audio: Union[np.ndarray, mx.array],
    model: SileroVAD,
    *,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> List[dict]:
    if sampling_rate != model.config.sample_rate:
        raise ValueError(f"sampling_rate must be {model.config.sample_rate}")

    if isinstance(audio, mx.array):
        audio_np = np.array(audio).astype(np.float32)
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)

    chunk_size = model.config.chunk_size
    pad_amount = chunk_size - (len(audio_np) % chunk_size)
    if pad_amount and pad_amount < chunk_size:
        audio_np = np.concatenate([audio_np, np.zeros(pad_amount, dtype=np.float32)])

    n_chunks = len(audio_np) // chunk_size
    chunks = audio_np.reshape(n_chunks, chunk_size)

    H = model.config.lstm_hidden_size
    ctx_size = model.config.context_size
    h = mx.zeros((1, H))
    c = mx.zeros((1, H))
    context = mx.zeros((1, ctx_size))

    probs = np.zeros(n_chunks, dtype=np.float32)
    eval_every = 32
    pending = []
    for i in range(n_chunks):
        chunk = mx.array(chunks[i : i + 1])
        prob, h, c, context = model(chunk, h, c, context)
        pending.append(prob)
        if (i + 1) % eval_every == 0 or i == n_chunks - 1:
            mx.eval(*pending, h, c, context)
            for j, p in enumerate(pending):
                probs[i - len(pending) + 1 + j] = float(p[0])
            pending = []

    speech_pad_chunks = max(0, int(speech_pad_ms / 1000 * sampling_rate / chunk_size))
    min_speech_chunks = max(1, int(min_speech_duration_ms / 1000 * sampling_rate / chunk_size))
    min_silence_chunks = max(1, int(min_silence_duration_ms / 1000 * sampling_rate / chunk_size))

    segments = []
    in_speech = False
    seg_start = 0
    silent_run = 0
    for idx, p in enumerate(probs):
        if p >= threshold:
            if not in_speech:
                seg_start = max(0, idx - speech_pad_chunks)
                in_speech = True
            silent_run = 0
        else:
            if in_speech:
                silent_run += 1
                if silent_run >= min_silence_chunks:
                    seg_end = idx - silent_run + speech_pad_chunks
                    if seg_end - seg_start >= min_speech_chunks:
                        segments.append((seg_start, seg_end))
                    in_speech = False
                    silent_run = 0
    if in_speech:
        seg_end = n_chunks
        if seg_end - seg_start >= min_speech_chunks:
            segments.append((seg_start, seg_end))

    return [
        {
            "start": s * chunk_size / sampling_rate,
            "end": e * chunk_size / sampling_rate,
            "start_sample": s * chunk_size,
            "end_sample": e * chunk_size,
        }
        for s, e in segments
    ]
