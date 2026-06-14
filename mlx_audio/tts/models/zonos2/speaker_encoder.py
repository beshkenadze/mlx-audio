"""ZONOS2 voice (speaker) encoder: ref-wav -> 2048-D x-vector speaker embedding.

The HF checkpoint ``marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`` referenced by
the upstream loader is — despite its name — an **ECAPA-TDNN x-vector speaker
encoder**, not a Qwen3 transformer. Its on-disk ``config.json`` declares
``architectures: ["EcapaTdnnSpeakerEncoder"]`` / ``model_type:
"ecapa_tdnn_speaker_encoder"`` with ``mel_dim=128`` -> ``enc_dim=2048`` and the
weight tensors are plain ``Conv1d`` blocks (``blocks.*``, ``mfa``, ``asp``,
``fc``) with no transformer / no BatchNorm. The module name is the source of the
"Qwen3" misnomer; the shipped architecture is the ground truth.

This module therefore wires the (unchanged) log-mel ``VoiceMelFrontend`` into the
shared MLX ``EcapaTdnnBackbone`` (``mlx_audio.codec.models.ecapa_tdnn``) and
returns a *pooled* utterance-level x-vector ``[B, 2048]`` (NOT a per-frame
sequence). The class is still named ``Qwen3VoiceEmbedding`` for API stability.

Frontend reference (ground truth), matching the checkpoint's
``feature_extraction_ecapa_tdnn.py`` exactly:
  reflect-pad the waveform by ``(n_fft - hop_length)//2`` on both sides, STFT with
  a periodic Hann window (``win_length = n_fft``, ``center=False``), magnitude
  spectrum projected through a slaney mel filterbank (slaney norm + slaney mel
  scale), ``log(clamp(mel, min=1e-5))``, transposed to ``[B, frames, 128]``.
  SR 24000, n_fft 1024, hop 256, n_mels 128, fmin 0, fmax 12000.

Backbone reference (ground truth): the checkpoint's ``modeling_ecapa_tdnn.py``
``EcapaTdnnSpeakerEncoder.forward(input_values=mel)`` — block0 (TDNN) + 3
SE-Res2Net blocks + MFA + attentive-statistics pooling + a final ``fc`` conv ->
``[B, enc_dim]``. There is no BatchNorm in the checkpoint; the shared
``EcapaTdnnBackbone`` carries BatchNorm layers which the weight converter
neutralises to identity (a uniform, cosine-invariant scale), so the reused module
reproduces the checkpoint to high fidelity.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.codec.models.ecapa_tdnn import EcapaTdnnBackbone, EcapaTdnnConfig
from mlx_audio.dsp import hanning, mel_filters, stft

from .config import ZONOS2Config

# Frontend constants — mirror EcapaTdnnFeatureExtractor / Qwen3SpeakerEmbedding.
_TARGET_SAMPLE_RATE = 24_000
_N_FFT = 1024
_HOP_LENGTH = 256
_WIN_LENGTH = 1024
_N_MELS = 128
_F_MIN = 0.0
_F_MAX = 12_000.0
_LOG_CLAMP_MIN = 1e-5

# ECAPA-TDNN encoder geometry (checkpoint config.json: enc_* fields).
# enc_channels = [512, 512, 512, 512, 1536]: block0 (512) + 3 SE-Res2Net blocks
# (512 each) feed the MFA over their concatenation (3*512 = 1536). The shared
# EcapaTdnnBackbone derives mfa/asp/fc widths from ``channels`` (= 512 here); the
# embed_dim (checkpoint enc_dim 2048) is taken from config.speaker_embedding_dim.
_ENC_CHANNELS = 512
_ENC_KERNEL_SIZES = [5, 3, 3, 3, 1]
_ENC_DILATIONS = [1, 2, 3, 4, 1]
_ENC_RES2NET_SCALE = 8
_ENC_SE_CHANNELS = 128
_ENC_ATTENTION_CHANNELS = 128


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
    """Log-mel frontend matching ``EcapaTdnnFeatureExtractor._compute_mel``.

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


def ecapa_config_from_zonos2(config: ZONOS2Config) -> EcapaTdnnConfig:
    """Build the shared ``EcapaTdnnConfig`` for the ZONOS2 speaker encoder.

    The widths come from the checkpoint's ``config.json`` (``enc_*`` fields). The
    embedding width tracks ``config.speaker_embedding_dim`` (2048 for the released
    checkpoint) so the pooled x-vector matches the conditioning-stack contract.
    ``global_context=True`` because the reference ASP always concatenates
    ``[x, mean, std]`` before the attention TDNN. ``conv_padding_mode="reflect"``
    matches the checkpoint's ``padding="same", padding_mode="reflect"`` convs
    (without it the conv edge frames diverge: cosine ~0.988 vs ~1.000).
    """
    return EcapaTdnnConfig(
        input_size=_N_MELS,
        channels=_ENC_CHANNELS,
        embed_dim=config.speaker_embedding_dim,
        kernel_sizes=list(_ENC_KERNEL_SIZES),
        dilations=list(_ENC_DILATIONS),
        attention_channels=_ENC_ATTENTION_CHANNELS,
        res2net_scale=_ENC_RES2NET_SCALE,
        se_channels=_ENC_SE_CHANNELS,
        global_context=True,
        conv_padding_mode="reflect",
    )


class Qwen3VoiceEmbedding(nn.Module):
    """Mel frontend + ECAPA-TDNN backbone -> pooled ``[B, 2048]`` x-vector.

    Despite the legacy class name (kept for API stability — the HF repo is named
    "Qwen3-Voice-Embedding"), the backbone is an ECAPA-TDNN speaker encoder, the
    architecture actually shipped in the checkpoint. ``__call__`` returns a single
    utterance-level speaker embedding per reference clip, NOT a per-frame sequence.
    """

    def __init__(self, config: ZONOS2Config) -> None:
        super().__init__()
        self.config = config
        self.target_sample_rate = _TARGET_SAMPLE_RATE
        self.frontend = VoiceMelFrontend(config)
        self.embed_dim = config.speaker_embedding_dim
        self.backbone = EcapaTdnnBackbone(ecapa_config_from_zonos2(config))

    def __call__(self, wav: mx.array, sample_rate: int) -> mx.array:
        """Reference waveform -> pooled ``[B, embed_dim]`` x-vector.

        Matches upstream ``Qwen3SpeakerEmbedding.prepare_input`` semantics: a 2-D
        input is a single multi-channel clip ``[C, T]`` and is downmixed to mono
        by averaging channels (NOT a batch of clips). One reference clip in, one
        ``[1, embed_dim]`` speaker embedding out.

        Args:
            wav: ``[T]`` mono or ``[C, T]`` multi-channel waveform.
            sample_rate: Sample rate of ``wav``; resampled to 24 kHz if different.
        """
        wav = self._prepare_input(wav, sample_rate)
        mel = self.frontend(wav)  # [1, frames, n_mels]
        return self.backbone(mel)  # [1, embed_dim]

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
