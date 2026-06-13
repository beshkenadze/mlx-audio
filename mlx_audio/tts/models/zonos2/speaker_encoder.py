"""ZONOS2 voice (speaker) encoder: ref-wav -> 2048-D speaker features.

Mirrors the upstream ``zonos2.models.speaker_cloning.Qwen3SpeakerEmbedding``
preprocessing (``_make_mel`` / ``mel_transform``) and wraps a Qwen3 transformer
backbone that consumes the log-mel features and returns a per-frame hidden state
of width ``2048`` (the ``last_hidden_state`` the conditioning stack expects).

Upstream reference (ground truth):
  ``python/zonos2/models/speaker_cloning.py`` -> ``Qwen3SpeakerEmbedding`` which
  builds the mel with ``torchaudio.transforms.MelSpectrogram(sample_rate=24000,
  n_fft=1024, win_length=1024, hop_length=256, n_mels=128, f_min=0, f_max=12000,
  power=1.0, center=False, norm="slaney", mel_scale="slaney")``, reflect-pads the
  waveform by ``(n_fft - hop_length)//2`` on each side, then
  ``log(clamp(mel, min=1e-5))`` and transposes to ``[B, frames, 128]``.

IMPORTANT discrepancy (documented for the Phase-D coordinator):
  The HF repo ``marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`` referenced by the
  upstream loader is, by its on-disk ``config.json`` / ``modeling_ecapa_tdnn.py``,
  an **ECAPA-TDNN** x-vector encoder (``architectures: ["EcapaTdnnSpeakerEncoder"]``,
  ``mel_dim=128`` -> ``enc_dim=2048``) whose ``last_hidden_state`` is a *pooled*
  ``[B, 2048]`` vector, not a per-frame ``[B, frames, 2048]`` sequence. The repo name
  ("Qwen3 ... 1.7B") does not match the shipped architecture. This module follows
  the ZONOS2 ``CONTRACT.md`` API (Qwen3 backbone -> ``[B, frames, 2048]``); whether
  the integration ultimately loads the ECAPA-TDNN checkpoint or a true Qwen3
  backbone is a Phase-D wiring decision. The mel frontend here is identical for
  both (the ECAPA feature extractor uses the exact same parameters).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelArgs
from mlx_lm.models.qwen3 import Qwen3Model

from mlx_audio.dsp import hanning, mel_filters, stft

from .config import ZONOS2Config

# Frontend constants — mirror Qwen3SpeakerEmbedding / EcapaTdnnFeatureExtractor.
_TARGET_SAMPLE_RATE = 24_000
_N_FFT = 1024
_HOP_LENGTH = 256
_WIN_LENGTH = 1024
_N_MELS = 128
_F_MIN = 0.0
_F_MAX = 12_000.0
_LOG_CLAMP_MIN = 1e-5


def _reflect_pad_1d(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad a 1-D signal by ``pad`` on each side (torch ``mode='reflect'``).

    The reflection excludes the boundary sample itself, matching
    ``torch.nn.functional.pad(..., mode="reflect")`` and ``dsp.stft``'s reflect pad.
    """
    if pad <= 0:
        return x
    if x.shape[0] <= pad:
        # torch.nn.functional.pad(mode="reflect") requires pad < input length;
        # raise loudly instead of silently building a too-short reflection.
        raise ValueError(
            f"waveform too short for reflect pad: length {x.shape[0]} <= pad {pad}"
        )
    prefix = x[1 : pad + 1][::-1]
    suffix = x[-(pad + 1) : -1][::-1]
    return mx.concatenate([prefix, x, suffix])


class VoiceMelFrontend(nn.Module):
    """Log-mel frontend matching ``Qwen3SpeakerEmbedding._make_mel``.

    Produces ``[B, frames, 128]`` magnitude log-mel features:
      * reflect-pad the waveform by ``(n_fft - hop_length)//2`` on both sides;
      * STFT with a periodic Hann window, ``win_length = n_fft``, ``center=False``;
      * magnitude spectrum (``power=1.0``) projected through a slaney mel filterbank
        (slaney norm + slaney mel scale);
      * ``log(clamp(mel, min=1e-5))``;
      * transpose to ``[B, frames, n_mels]``.
    """

    def __init__(self, config: ZONOS2Config) -> None:
        super().__init__()
        self.sample_rate = _TARGET_SAMPLE_RATE
        self.n_fft = _N_FFT
        self.hop_length = _HOP_LENGTH
        self.win_length = _WIN_LENGTH
        self.n_mels = _N_MELS
        self.f_min = _F_MIN
        self.f_max = _F_MAX
        self.pad = (self.n_fft - self.hop_length) // 2
        # Periodic Hann window (torchaudio MelSpectrogram / torch.hann_window default).
        self._window = hanning(self.win_length, periodic=True)
        # Slaney triangular filterbank, shape (n_mels, n_fft // 2 + 1).
        self._mel_basis = mel_filters(
            self.sample_rate,
            self.n_fft,
            self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            norm="slaney",
            mel_scale="slaney",
        )

    def _make_mel(self, wav: mx.array) -> mx.array:
        """Single-channel waveform ``[T]`` -> log-mel ``[frames, n_mels]``."""
        wav = _reflect_pad_1d(wav, self.pad)
        spec = stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window,
            center=False,
            pad_mode="reflect",
        )
        # power=1.0 -> magnitude spectrum; spec is [frames, n_fft // 2 + 1].
        magnitude = mx.abs(spec)
        mel = self._mel_basis.astype(magnitude.dtype) @ magnitude.T
        mel = mx.log(mx.clip(mel, _LOG_CLAMP_MIN, None))
        return mel.T

    def __call__(self, wav: mx.array) -> mx.array:
        """Batched waveform ``[B, T]`` -> log-mel ``[B, frames, n_mels]``."""
        if wav.ndim == 1:
            wav = wav[None, :]
        mels = [self._make_mel(wav[b]) for b in range(wav.shape[0])]
        return mx.stack(mels, axis=0)


def _resample_linear(wav: mx.array, orig_sr: int, target_sr: int) -> mx.array:
    """Lightweight linear resample of a single channel ``[T]`` to ``target_sr``.

    The upstream uses ``torchaudio.transforms.Resample`` (sinc). Linear resampling
    is a dependency-free approximation used only when the input rate differs from
    24 kHz; pass already-24 kHz audio for exact parity.
    """
    if orig_sr == target_sr:
        return wav
    n_in = wav.shape[0]
    n_out = max(1, int(round(n_in * target_sr / orig_sr)))
    # Sample positions in the input grid for each output index.
    positions = mx.arange(n_out, dtype=mx.float32) * (orig_sr / target_sr)
    left = mx.floor(positions).astype(mx.int32)
    left = mx.clip(left, 0, n_in - 1)
    right = mx.clip(left + 1, 0, n_in - 1)
    frac = positions - left.astype(mx.float32)
    return wav[left] * (1.0 - frac) + wav[right] * frac


def _qwen3_args_from_config(config: ZONOS2Config) -> Qwen3ModelArgs:
    """Build Qwen3 backbone args sized to the ZONOS2 speaker-embedding width."""
    return Qwen3ModelArgs(
        model_type="qwen3",
        hidden_size=config.speaker_embedding_dim,
        num_hidden_layers=config.n_layers,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_qo_heads,
        rms_norm_eps=config.norm_eps,
        # vocab_size is only used for the (unused) token embedding table; keep small.
        vocab_size=1,
        num_key_value_heads=config.n_kv_heads,
        max_position_embeddings=config.max_seqlen,
        rope_theta=config.rope_theta,
        head_dim=config.head_dim,
        tie_word_embeddings=True,
    )


class Qwen3VoiceEmbedding(nn.Module):
    """Mel frontend + Qwen3 backbone -> per-frame ``[1, frames, 2048]`` features.

    The backbone consumes the ``[1, frames, n_mels]`` log-mel via an input
    projection (``n_mels`` -> ``hidden``) fed as ``input_embeddings`` so the
    Qwen3 transformer runs over the mel frames, returning ``last_hidden_state``
    ``[1, frames, hidden]`` (``hidden == speaker_embedding_dim``, 2048 for the
    1.7B config). One reference clip per call. See the module docstring for the
    upstream-checkpoint caveat.
    """

    def __init__(self, config: ZONOS2Config) -> None:
        super().__init__()
        self.config = config
        self.target_sample_rate = _TARGET_SAMPLE_RATE
        self.frontend = VoiceMelFrontend(config)
        args = _qwen3_args_from_config(config)
        self.hidden_size = args.hidden_size
        # mel (128) -> hidden projection feeding the transformer as input_embeddings.
        self.mel_proj = nn.Linear(_N_MELS, args.hidden_size, bias=False)
        self.backbone = Qwen3Model(args)

    def __call__(self, wav: mx.array, sample_rate: int) -> mx.array:
        """Reference waveform -> ``[1, frames, hidden]`` speaker features.

        Matches upstream ``Qwen3SpeakerEmbedding.prepare_input`` semantics: a 2-D
        input is a single multi-channel clip ``[C, T]`` and is downmixed to mono
        by averaging channels (NOT a batch of clips). One reference clip in, one
        ``[1, frames, hidden]`` feature sequence out.

        Args:
            wav: ``[T]`` mono or ``[C, T]`` multi-channel waveform.
            sample_rate: Sample rate of ``wav``; resampled to 24 kHz if different.
        """
        wav = self._prepare_input(wav, sample_rate)
        mel = self.frontend(wav)
        h = self.mel_proj(mel)
        # NOTE: Qwen3Model applies a causal attention mask over the mel frames, so
        # frame t only attends to <= t. A bidirectional pass is closer to a true
        # utterance-level speaker encoder; reconciling the backbone choice (and the
        # ECAPA-TDNN checkpoint discrepancy noted in the module docstring) is a
        # Phase-D coordinator decision.
        return self.backbone(None, cache=None, input_embeddings=h)

    def _prepare_input(self, wav: mx.array, sample_rate: int) -> mx.array:
        """Downmix to mono, resample to 24 kHz, return batched ``[1, T]``."""
        if wav.ndim > 2:
            raise ValueError(f"wav must be 1-D or 2-D, got shape {wav.shape}")
        if wav.ndim == 2:
            # [C, T] multi-channel clip -> mono (mean over channels).
            wav = wav.mean(axis=0)
        wav = wav.astype(mx.float32)
        if sample_rate != self.target_sample_rate:
            wav = _resample_linear(wav, sample_rate, self.target_sample_rate)
        return wav[None, :]
