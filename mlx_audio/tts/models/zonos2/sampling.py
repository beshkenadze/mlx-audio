"""Per-codebook sampling for the ZONOS2 MLX port.

Mirrors the sampling semantics of upstream ``python/zonos2/tts/sampler.py``
(``sample_tts``): optional logit softcap, then **repetition penalty**, temperature
scaling, top-k, softmax, top-p and **min-p** filtering, with greedy decoding when
``temperature <= 0``. Each audio codebook is sampled independently along the last
(vocab) axis, so a single sampler handles both ``[..., n_codebooks, vocab]`` and
``[..., vocab]`` logits.

Upstream ``sample_tts`` order (after the model has already soft-capped the head):

    repetition_penalty(logits) -> temperature scale -> top_k(mask -> -inf)
    -> softmax -> top_p(probs) -> min_p(probs) -> multinomial

We reproduce that exact order and chain so the sampled distribution matches the
CUDA reference. Each filter (rep-penalty, top-k, top-p, min-p) is implemented
directly against the upstream ``apply_*`` helpers; the final categorical draw
uses ``mx.random.categorical`` on the log of the filtered probabilities.

Reference: https://github.com/Zyphra/ZONOS2 -> python/zonos2/tts/sampler.py
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx


def softcap_logits(logits: mx.array, cap: float) -> mx.array:
    """Squash logits into ``(-cap, cap)`` via ``cap * tanh(logits / cap)``.

    Mirrors the ZONOS2 ``loss_softcap`` applied to the output head before
    sampling. A non-positive ``cap`` disables the transform (identity).
    """
    if cap <= 0.0:
        return logits
    return cap * mx.tanh(logits / cap)


def apply_repetition_penalty(
    logits: mx.array,
    repetition_token_ids: Optional[mx.array],
    repetition_penalty: float,
    *,
    codebook_size: Optional[int] = None,
    repetition_codebooks: Optional[int] = None,
) -> mx.array:
    """Penalize logits for tokens recently seen in the SAME codebook.

    Mirrors upstream ``apply_repetition_penalty``: a token id seen in the rolling
    window of codebook ``c`` penalizes ONLY codebook ``c``'s logits (positive
    logits are divided by the penalty, negative logits multiplied). The penalty
    is a flat factor — it does not scale with how many times a token recurred.

    Args:
        logits: ``[..., C, V]`` per-codebook logits.
        repetition_token_ids: ``[..., C, window]`` int ids of the recent frames
            (per codebook). Entries ``< 0`` (or out of ``[0, V)``) are ignored.
            ``None`` / empty disables the penalty.
        repetition_penalty: factor ``>= 1.0``; ``<= 1.0`` disables the penalty.
        codebook_size: if given, ids ``>= codebook_size`` (eoa / pad) are treated
            as invalid before building the penalty mask (matches upstream's
            ``valid = (history >= 0) & (history < codebook_size)``).
        repetition_codebooks: if given, only the first ``repetition_codebooks``
            codebook rows receive the penalty (upstream zeroes the rest of the
            history mask). ``None`` penalizes every codebook.

    Returns:
        The penalized ``logits`` (same shape / dtype).
    """
    if repetition_token_ids is None or repetition_penalty <= 1.0:
        return logits
    if repetition_token_ids.size == 0:
        return logits

    V = logits.shape[-1]
    C = logits.shape[-2]
    ids = repetition_token_ids.astype(mx.int32)

    # An id contributes to the penalty mask only if it indexes a real codebook
    # entry (upstream masks eoa/pad via ``< codebook_size`` and clamps the rest).
    valid = ids >= 0
    if codebook_size is not None:
        valid = valid & (ids < int(codebook_size))
    else:
        valid = valid & (ids < V)

    # Restrict the penalty to the first ``repetition_codebooks`` codebooks
    # (upstream: ``valid[repetition_codebooks:] = False``).
    if repetition_codebooks is not None and repetition_codebooks < C:
        keep = mx.arange(C).reshape((1,) * (ids.ndim - 2) + (C, 1)) < int(
            repetition_codebooks
        )
        valid = valid & keep

    safe_ids = mx.clip(ids, 0, V - 1)
    # ``repeated[..., c, v]`` is True iff token ``v`` appears (as a valid entry)
    # in codebook ``c``'s window. Broadcast the window ids against ``arange(V)``
    # and reduce over the window axis (``any`` == upstream's ``counts > 0``).
    vocab_axis = mx.arange(V).reshape((1,) * ids.ndim + (V,))
    matches = (safe_ids[..., None] == vocab_axis) & valid[..., None]
    repeated = mx.any(matches, axis=-2)  # [..., C, V]

    penalty = max(float(repetition_penalty), 1.0)
    adjusted = mx.where(logits > 0, logits / penalty, logits * penalty)
    return mx.where(repeated, adjusted, logits)


def apply_top_p(probs: mx.array, p: float) -> mx.array:
    """Nucleus (top-p) filter on probabilities (upstream ``apply_top_p``).

    Keeps the smallest set of highest-probability tokens whose cumulative mass
    reaches ``p`` (upstream rule: drop where ``cumsum - prob > p``), then
    renormalizes. ``p <= 0`` or ``p >= 1`` is a no-op.
    """
    if p <= 0.0 or p >= 1.0:
        return probs
    order = mx.argsort(-probs, axis=-1)
    probs_sort = mx.take_along_axis(probs, order, axis=-1)
    probs_sum = mx.cumsum(probs_sort, axis=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort = mx.where(mask, mx.array(0.0, dtype=probs.dtype), probs_sort)
    # Scatter the filtered sorted probs back to their original positions.
    inv = mx.argsort(order, axis=-1)
    probs = mx.take_along_axis(probs_sort, inv, axis=-1)
    denom = mx.maximum(mx.sum(probs, axis=-1, keepdims=True), 1e-8)
    return probs / denom


def apply_min_p(probs: mx.array, min_p: float) -> mx.array:
    """Drop probs ``< min_p * max_prob`` then renormalize (upstream ``apply_min_p``).

    ``min_p <= 0`` is a no-op. The renormalization clamps the denominator at
    ``1e-8`` like upstream; a fully-masked row stays all-zero (the caller's
    argmax fallback handles it).
    """
    if min_p <= 0.0:
        return probs
    top = mx.max(probs, axis=-1, keepdims=True)
    keep = probs >= (min_p * top)
    filtered = mx.where(keep, probs, mx.array(0.0, dtype=probs.dtype))
    denom = mx.maximum(mx.sum(filtered, axis=-1, keepdims=True), 1e-8)
    return filtered / denom


def _apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    """Mask all but the ``top_k`` highest logits to ``-inf`` (upstream top-k).

    Upstream takes ``topk`` values, finds the ``kth`` (smallest kept) and masks
    ``logits < kth``. ``top_k <= 0`` or ``>= vocab`` disables the filter.
    """
    V = logits.shape[-1]
    if not (0 < top_k < V):
        return logits
    kth = mx.topk(logits, top_k, axis=-1)[..., :1]  # smallest of the top_k
    neg_inf = mx.array(float("-inf"), dtype=logits.dtype)
    return mx.where(logits < kth, neg_inf, logits)


def sample_codebook_logits(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_token_ids: Optional[mx.array] = None,
    repetition_penalty: float = 1.0,
    repetition_codebooks: Optional[int] = None,
    codebook_size: Optional[int] = None,
) -> mx.array:
    """Sample one token per codebook, mirroring upstream ``sample_tts``.

    Order (matching upstream): repetition_penalty -> temperature -> top_k
    -> softmax -> top_p -> min_p -> categorical. A fully-masked row falls back
    to the temperature-scaled argmax (upstream's invalid-row greedy fallback).
    """
    logits = apply_repetition_penalty(
        logits,
        repetition_token_ids,
        repetition_penalty,
        codebook_size=codebook_size,
        repetition_codebooks=repetition_codebooks,
    )

    scaled = logits / max(temperature, 1e-8)
    scaled = _apply_top_k(scaled, top_k)
    probs = mx.softmax(scaled, axis=-1)
    probs = apply_top_p(probs, top_p)
    probs = apply_min_p(probs, min_p)

    # Fully-masked rows (aggressive min_p/top_p) would make the categorical draw
    # ill-defined; fall back to greedy argmax for those rows like upstream.
    row_sum = mx.sum(probs, axis=-1, keepdims=True)
    invalid = row_sum <= 0.0
    greedy = mx.argmax(scaled, axis=-1)
    # log(probs) gives -inf for zeroed tokens, masking them in categorical.
    logp = mx.log(mx.where(probs > 0, probs, mx.array(1e-38, dtype=probs.dtype)))
    sampled = mx.random.categorical(logp, axis=-1)
    return mx.where(invalid[..., 0], greedy, sampled).astype(mx.int32)


def make_codebook_sampler(
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
    softcap: float = 0.0,
    *,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    repetition_codebooks: Optional[int] = None,
    codebook_size: Optional[int] = None,
) -> Callable[..., mx.array]:
    """Build a per-codebook sampler matching upstream ``sample_tts`` semantics.

    The returned ``sampler(logits, repetition_token_ids=None)`` accepts logits
    shaped ``[..., vocab]`` or ``[..., n_codebooks, vocab]`` and returns
    ``int32`` token ids with the trailing vocab axis removed.

    Args:
        temperature: Softmax temperature. ``<= 0`` selects greedy (``argmax``).
        top_k: Keep only the ``top_k`` highest-probability tokens (``0`` = off).
            Values ``>= vocab`` disable the filter (matching upstream).
        top_p: Nucleus cutoff in ``(0, 1)``; ``1.0`` (or ``0.0``) disables it.
        softcap: Logit softcap ``cap``; applied first when ``> 0``.
        min_p: Min-p cutoff; tokens with prob ``< min_p * max_prob`` are dropped.
            ``0`` disables it.
        repetition_penalty: factor ``>= 1.0`` applied to per-codebook logits for
            tokens seen in the recent-token window passed at call time. ``<= 1.0``
            disables it.
        repetition_codebooks: only the first N codebooks receive the rep penalty.
        codebook_size: real audio vocab (eoa/pad ids ``>=`` this are not counted
            as repeats), forwarded to :func:`apply_repetition_penalty`.

    Returns:
        ``sampler(logits, repetition_token_ids=None) -> mx.array``.

    Note:
        The greedy path (``temperature <= 0``) intentionally does NOT apply the
        repetition penalty, keeping the parity / greedy decode path byte-exact.
    """
    top_k = max(int(top_k), 0)
    rep_penalty = max(float(repetition_penalty), 1.0)
    min_p = max(float(min_p), 0.0)

    if temperature <= 0:
        # Greedy: argmax over vocab, independent per codebook/batch element.
        # softcap is monotonic so it does not change the argmax, but apply it
        # anyway to keep the two branches behaviourally identical. Repetition
        # penalty is skipped here (greedy decode stays parity-exact).
        def greedy_sampler(
            logits: mx.array, repetition_token_ids: Optional[mx.array] = None
        ) -> mx.array:
            logits = softcap_logits(logits, softcap)
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        return greedy_sampler

    def sampler(
        logits: mx.array, repetition_token_ids: Optional[mx.array] = None
    ) -> mx.array:
        logits = softcap_logits(logits, softcap)
        return sample_codebook_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_token_ids=repetition_token_ids,
            repetition_penalty=rep_penalty,
            repetition_codebooks=repetition_codebooks,
            codebook_size=codebook_size,
        )

    return sampler
