"""Synthetic shape/wiring tests for the ZONOS2 voice (speaker) encoder.

No weights and no GPU: verifies that the log-mel frontend matches the upstream
frame geometry and that the ECAPA-TDNN backbone wrapper pools the mel features
into a single utterance-level x-vector ``[B, embed_dim]``. Full numerical parity
vs the real checkpoint is covered by ``parity_test/zonos2/check_ecapa.py``.
"""

from __future__ import annotations

import math

import mlx.core as mx

from mlx_audio.tts.models.zonos2.config import ZONOS2Config
from mlx_audio.tts.models.zonos2.speaker_encoder import (
    Qwen3VoiceEmbedding,
    VoiceMelFrontend,
    ecapa_config_from_zonos2,
)

_SAMPLE_RATE = 24_000
_N_FFT = 1024
_HOP = 256
_N_MELS = 128


def _expected_frames(num_samples: int) -> int:
    """Upstream frame count: reflect-pad both sides, center=False STFT."""
    pad = (_N_FFT - _HOP) // 2
    padded = num_samples + 2 * pad
    return 1 + (padded - _N_FFT) // _HOP


def _sine(num_samples: int, freq_hz: float = 220.0) -> mx.array:
    return mx.array(
        [
            math.sin(2 * math.pi * freq_hz * n / _SAMPLE_RATE)
            for n in range(num_samples)
        ],
        dtype=mx.float32,
    )


def _tiny_config(embed_dim: int = 2048) -> ZONOS2Config:
    """A small config for fast shape-flow testing (ECAPA geometry is fixed)."""
    return ZONOS2Config(
        n_layers=2,
        dim=32,
        head_dim=8,
        n_heads=4,
        n_kv_heads=2,
        speaker_embedding_dim=embed_dim,
        max_seqlen=4096,
        moe_n_experts=1,
    )


def test_mel_frontend_shape_and_frame_count():
    config = _tiny_config()
    frontend = VoiceMelFrontend(config)

    wav = _sine(_SAMPLE_RATE)[None, :]  # [1, 24000]
    mel = frontend(wav)

    expected = _expected_frames(_SAMPLE_RATE)
    assert mel.shape == (1, expected, _N_MELS)
    assert mel.dtype == mx.float32
    # Log-mel of a clean sine should be finite (no NaN/Inf from the log clamp).
    assert bool(mx.all(mx.isfinite(mel)))


def test_mel_frontend_accepts_unbatched_waveform():
    frontend = VoiceMelFrontend(_tiny_config())
    half_second = _SAMPLE_RATE // 2

    mel = frontend(_sine(half_second))  # 1-D input -> [1, frames, 128]

    assert mel.shape == (1, _expected_frames(half_second), _N_MELS)


def test_mel_frontend_rejects_too_short_waveform():
    frontend = VoiceMelFrontend(_tiny_config())
    pad = (_N_FFT - _HOP) // 2  # 384

    too_short = _sine(pad)  # length == pad -> reflect pad is invalid (torch raises)
    raised = False
    try:
        frontend(too_short)
    except ValueError:
        raised = True
    assert raised


def test_ecapa_config_matches_checkpoint_geometry():
    """The derived ECAPA config must match the checkpoint's enc_* fields."""
    cfg = ecapa_config_from_zonos2(_tiny_config(embed_dim=2048))
    assert cfg.input_size == _N_MELS
    assert cfg.channels == 512
    assert cfg.embed_dim == 2048
    assert cfg.kernel_sizes == [5, 3, 3, 3, 1]
    assert cfg.dilations == [1, 2, 3, 4, 1]
    assert cfg.res2net_scale == 8
    assert cfg.se_channels == 128
    assert cfg.attention_channels == 128
    assert cfg.global_context is True


def test_voice_embedding_pools_to_embed_dim():
    embed_dim = 2048
    config = _tiny_config(embed_dim=embed_dim)
    encoder = Qwen3VoiceEmbedding(config)

    wav = _sine(_SAMPLE_RATE)[None, :]  # [1, 24000] already at 24 kHz
    out = encoder(wav, sample_rate=_SAMPLE_RATE)

    # Pooled x-vector: one embedding per clip, NOT a per-frame sequence.
    assert out.shape == (1, embed_dim)
    assert bool(mx.all(mx.isfinite(out)))


def test_voice_embedding_resamples_non_target_rate():
    embed_dim = 2048
    encoder = Qwen3VoiceEmbedding(_tiny_config(embed_dim=embed_dim))

    src_sr = 44_100
    n_in = src_sr  # 1 second
    wav = mx.array(
        [math.sin(2 * math.pi * 220.0 * n / src_sr) for n in range(n_in)],
        dtype=mx.float32,
    )
    out = encoder(wav, sample_rate=src_sr)

    assert out.shape == (1, embed_dim)
    assert bool(mx.all(mx.isfinite(out)))

    # The pooled output hides the frame count, so assert the resample stage itself
    # actually changed the length (44.1 kHz -> 24 kHz) via the prepared waveform.
    prepared = encoder._prepare_input(wav, src_sr)
    n_resampled = max(1, round(n_in * _SAMPLE_RATE / src_sr))
    assert prepared.shape == (1, n_resampled)
    assert n_resampled != n_in


def test_voice_embedding_downmixes_stereo():
    embed_dim = 2048
    encoder = Qwen3VoiceEmbedding(_tiny_config(embed_dim=embed_dim))

    mono = _sine(_SAMPLE_RATE)
    stereo = mx.stack([mono, mono], axis=0)  # [2, 24000]
    out = encoder(stereo, sample_rate=_SAMPLE_RATE)

    assert out.shape == (1, embed_dim)
    # Channel-averaging a duplicated stereo clip must equal the mono embedding
    # (downmix is a mean over channels, NOT a batch of two clips).
    mono_out = encoder(mono, sample_rate=_SAMPLE_RATE)
    assert mono_out.shape == (1, embed_dim)
    assert bool(mx.allclose(out, mono_out, atol=1e-4))
