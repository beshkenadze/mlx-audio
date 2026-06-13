"""Speaker conditioning for the ZONOS2 MLX port.

Two pieces:

* :class:`SpeakerLDA` -- the affine LDA projection applied to the raw 2048-D
  speaker embedding, mirroring upstream ``models/speaker_lda.py``
  (``SpeakerLDAProjection``). It is a plain ``F.linear`` (weight + bias, **no**
  mean-centering), so it maps directly to an ``mlx.nn.Linear``. Checkpoint keys:
  ``speaker_lda_projection.{weight,bias}``.

* :func:`speaking_rate_bucket` / :func:`quality_bucket` -- map a continuous
  value to the discrete conditioning-bucket index, parsing the boundary strings
  stored on :class:`~mlx_audio.tts.models.zonos2.config.ZONOS2Config`
  (``speaking_rate_buckets`` like ``"8-11"`` / ``"40+"``; ``quality_buckets``
  like ``"-50--45.5"`` / ``"60+"`` / exact ``"0"``). The boundary semantics
  mirror upstream ``server/api_server.py`` (``_speaking_rate_bucket_for_rate``,
  ``_quality_bucket_for_value`` and their ``*_BUCKET_RE`` parsers).

Reference: https://github.com/Zyphra/ZONOS2 -> python/zonos2/models/speaker_lda.py
and python/zonos2/server/api_server.py (bucket parsing / value->index).
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.zonos2.config import ZONOS2Config

# --- bucket boundary regexes (mirror upstream server/api_server.py) ---
# Speaking-rate buckets only ever use non-negative values, e.g. "8-11" / "40+".
_SPEAKING_RATE_CLOSED_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_SPEAKING_RATE_OPEN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")

# Quality buckets may carry signed numbers, so the separating "-" can sit next
# to a leading "-" of the upper bound, e.g. "-50--45.5" -> low=-50, high=-45.5.
_QUALITY_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_QUALITY_EXACT_RE = re.compile(rf"^\s*({_QUALITY_NUMBER})\s*$")
_QUALITY_CLOSED_RE = re.compile(
    rf"^\s*({_QUALITY_NUMBER})\s*-\s*({_QUALITY_NUMBER})\s*$"
)
_QUALITY_OPEN_RE = re.compile(rf"^\s*({_QUALITY_NUMBER})\s*\+\s*$")


class SpeakerLDA(nn.Module):
    """Affine LDA projection over raw speaker embeddings.

    Mirrors upstream ``SpeakerLDAProjection`` (``F.linear(x, weight, bias)``):
    a plain affine map with bias and **no** mean-centering. MLX ``nn.Linear``
    stores ``weight`` as ``[output_dim, input_dim]`` and computes
    ``x @ weight.T + bias`` -- identical to the upstream ``F.linear`` layout, so
    the checkpoint tensors ``speaker_lda_projection.{weight,bias}`` load directly
    onto ``self.proj.{weight,bias}``.
    """

    def __init__(self, config: ZONOS2Config) -> None:
        super().__init__()
        input_dim = config.speaker_embedding_dim
        output_dim = config.speaker_lda_dim
        if output_dim is None:
            raise ValueError("config.speaker_lda_dim must be set for SpeakerLDA.")
        self.proj = nn.Linear(input_dim, output_dim, bias=True)

    def __call__(self, emb_2048: mx.array) -> mx.array:
        return self.proj(emb_2048)


def _parse_speaking_rate_bucket(spec: str) -> Tuple[float, Optional[float]]:
    """Parse one speaking-rate bucket string into ``(low, high)``.

    ``high`` is ``None`` for an open-ended bucket like ``"40+"``.
    """
    closed = _SPEAKING_RATE_CLOSED_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))
    open_ended = _SPEAKING_RATE_OPEN_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None
    raise ValueError(
        f"Invalid speaking-rate bucket {spec!r}; expected ranges like '0-3' or '60+'."
    )


def _parse_quality_bucket(spec: str) -> Tuple[str, float, Optional[float]]:
    """Parse one quality bucket string into ``(kind, low, high)``.

    ``kind`` is ``"exact"`` (single value, ``high`` is ``None``) or ``"range"``
    (``high`` is ``None`` only for an open-ended ``"X+"`` bucket).
    """
    value = str(spec)
    exact = _QUALITY_EXACT_RE.match(value)
    if exact is not None:
        return "exact", float(exact.group(1)), None
    closed = _QUALITY_CLOSED_RE.match(value)
    if closed is not None:
        return "range", float(closed.group(1)), float(closed.group(2))
    open_ended = _QUALITY_OPEN_RE.match(value)
    if open_ended is not None:
        return "range", float(open_ended.group(1)), None
    raise ValueError(
        f"Invalid quality bucket {spec!r}; expected exact values like '0', "
        "ranges like '-30--25', or open-ended ranges like '22050+'."
    )


def speaking_rate_bucket(value: float, config: ZONOS2Config) -> int:
    """Return the speaking-rate bucket index for ``value``.

    Mirrors upstream ``_speaking_rate_bucket_for_rate`` (with ranges present):
    iterate the contiguous ``config.speaking_rate_buckets`` ranges in order and
    return the first whose upper bound is open (``"X+"``) or strictly greater
    than ``value`` (the boundary ``value == high`` falls into the *next*
    bucket). Falls back to the last bucket.
    """
    rate = float(value)
    if rate <= 0:
        raise ValueError("speaking_rate must be positive.")

    ranges = [
        _parse_speaking_rate_bucket(spec) for spec in config.speaking_rate_buckets
    ]
    if not ranges:
        raise ValueError("config.speaking_rate_buckets is empty.")

    for idx, (_, high) in enumerate(ranges):
        if high is None or (
            rate < high and not math.isclose(rate, high, rel_tol=1e-12, abs_tol=1e-9)
        ):
            return idx
    return len(ranges) - 1


def quality_bucket(feature: str, value: float, config: ZONOS2Config) -> int:
    """Return the quality bucket index for ``feature`` at ``value``.

    Mirrors upstream ``_quality_bucket_for_value``:

    * exact buckets win first when ``value`` matches (``math.isclose``);
    * among range buckets, an open-ended ``"X+"`` matches ``value >= low``, the
      last range matches inclusive ``low <= value <= high``, and every other
      range matches half-open ``low <= value < high``;
    * out-of-range values clamp to the first/last range bucket.
    """
    buckets = (config.quality_buckets or {}).get(feature)
    if not buckets:
        raise ValueError(
            f"config.quality_buckets has no entry for feature {feature!r}."
        )

    val = float(value)
    if not math.isfinite(val):
        raise ValueError("quality value must be finite.")

    specs = [_parse_quality_bucket(spec) for spec in buckets]

    for idx, (kind, low, _high) in enumerate(specs):
        if kind == "exact" and math.isclose(val, low, rel_tol=1e-12, abs_tol=1e-9):
            return idx

    range_indexes: List[int] = [
        idx for idx, (kind, _, _) in enumerate(specs) if kind == "range"
    ]
    if not range_indexes:
        raise ValueError(
            f"config.quality_buckets[{feature!r}] has no range buckets to place {val}."
        )

    for idx in range_indexes:
        _, low, high = specs[idx]
        if high is None:
            if val >= low:
                return idx
        elif idx == range_indexes[-1]:
            if low <= val <= high:
                return idx
        elif low <= val < high:
            return idx

    _, first_low, _ = specs[range_indexes[0]]
    if val < first_low:
        return range_indexes[0]
    return range_indexes[-1]
