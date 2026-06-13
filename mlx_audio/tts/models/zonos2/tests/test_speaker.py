"""Synthetic unit tests for the ZONOS2 speaker module.

No GPU/CUDA fixtures: ``SpeakerLDA`` is checked for shape only, and the bucket
helpers are checked against hand-derived boundary cases that mirror upstream
``server/api_server.py`` semantics.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_audio.tts.models.zonos2.config import ZONOS2Config
from mlx_audio.tts.models.zonos2.speaker import (
    SpeakerLDA,
    quality_bucket,
    speaking_rate_bucket,
)


def _config() -> ZONOS2Config:
    return ZONOS2Config(
        speaker_embedding_dim=2048,
        speaker_lda_dim=1024,
        # Contiguous, non-negative, ends open-ended (mirrors upstream rules).
        speaking_rate_buckets=("0-8", "8-11", "11-40", "40+"),
        quality_features=("lufs", "snr"),
        quality_buckets={
            # Fully-negative ranges + a closed positive range + open-ended.
            "lufs": ("-1000--50", "-50--45.5", "-45.5--40", "-40+"),
            # Includes an exact-value bucket and a fully-negative range.
            "snr": ("-1000-0", "0", "0-10", "10+"),
        },
    )


# --- SpeakerLDA --------------------------------------------------------------


def test_speaker_lda_shape():
    config = _config()
    lda = SpeakerLDA(config)
    out = lda(mx.zeros((2, 2048)))
    assert out.shape == (2, 1024)


def test_speaker_lda_is_affine_with_bias():
    config = _config()
    lda = SpeakerLDA(config)
    # Weight stored as [out, in]; bias present (mirrors F.linear, no centering).
    assert lda.proj.weight.shape == (1024, 2048)
    assert "bias" in lda.proj
    assert lda.proj.bias.shape == (1024,)


# --- speaking_rate_bucket ----------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.5, 0),  # small positive -> bucket 0 ("0-8")
        (8.0, 1),  # boundary value falls into the NEXT bucket ("8-11")
        (9.0, 1),  # inside "8-11"
        (11.0, 2),  # boundary -> "11-40"
        (39.9, 2),  # inside "11-40"
        (40.0, 3),  # boundary -> open-ended "40+"
        (100.0, 3),  # large -> last "40+"
    ],
)
def test_speaking_rate_bucket(value, expected):
    assert speaking_rate_bucket(value, _config()) == expected


def test_speaking_rate_bucket_rejects_non_positive():
    with pytest.raises(ValueError):
        speaking_rate_bucket(0.0, _config())


# --- quality_bucket ----------------------------------------------------------


def test_quality_bucket_negative_range():
    # -47 lands in the fully-negative range "-50--45.5" (index 1).
    assert quality_bucket("lufs", -47.0, _config()) == 1


@pytest.mark.parametrize(
    "value,expected",
    [
        (-500.0, 0),  # "-1000--50"
        (-50.0, 1),  # boundary -> half-open next "-50--45.5"
        (-45.5, 2),  # boundary -> "-45.5--40"
        (-40.0, 3),  # open-ended "-40+"
        (1000.0, 3),  # large -> open-ended bucket
    ],
)
def test_quality_bucket_lufs_boundaries(value, expected):
    assert quality_bucket("lufs", value, _config()) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (-5.0, 0),  # fully-negative range "-1000-0"
        (0.0, 1),  # exact bucket "0" wins over the range edges
        (5.0, 2),  # "0-10"
        (10.0, 3),  # open-ended "10+"
        (50.0, 3),  # large -> "10+"
    ],
)
def test_quality_bucket_snr_with_exact(value, expected):
    assert quality_bucket("snr", value, _config()) == expected


def test_quality_bucket_clamps_below_first_range():
    # Below the first range's lower bound -> clamp to first range bucket.
    assert quality_bucket("lufs", -5000.0, _config()) == 0


def test_quality_bucket_unknown_feature_raises():
    with pytest.raises(ValueError):
        quality_bucket("does_not_exist", 0.0, _config())
