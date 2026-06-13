"""Synthetic tests for the ZONOS2 per-codebook sampler.

No GPU / CUDA fixtures: pure shape + semantics checks on random logits.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.zonos2.sampling import (
    apply_min_p,
    apply_repetition_penalty,
    make_codebook_sampler,
    softcap_logits,
)


def test_temperature_zero_is_argmax() -> None:
    """temperature=0 must be deterministic argmax of the logits."""
    mx.random.seed(0)
    logits = mx.random.normal(shape=(2, 9, 1026))
    sampler = make_codebook_sampler(temperature=0.0)
    out = sampler(logits)
    expected = mx.argmax(logits, axis=-1)
    assert out.shape == (2, 9)
    assert mx.array_equal(out, expected)
    assert out.dtype == mx.int32


def test_negative_temperature_is_greedy() -> None:
    """temperature <= 0 (incl. negatives) falls back to argmax, never inverts."""
    mx.random.seed(1)
    logits = mx.random.normal(shape=(3, 1026))
    sampler = make_codebook_sampler(temperature=-1.0)
    assert mx.array_equal(sampler(logits), mx.argmax(logits, axis=-1))


def test_top_k_one_forces_argmax_even_at_high_temperature() -> None:
    """top_k=1 collapses the distribution to the argmax token deterministically."""
    mx.random.seed(2)
    logits = mx.random.normal(shape=(4, 9, 1026))
    expected = mx.argmax(logits, axis=-1)
    sampler = make_codebook_sampler(temperature=2.0, top_k=1)
    # Repeat across seeds: top_k=1 must always return the argmax token.
    for seed in range(5):
        mx.random.seed(seed)
        out = sampler(logits)
        assert out.shape == (4, 9)
        assert mx.array_equal(out, expected)


def test_softcap_maps_large_logit_through_cap_tanh() -> None:
    """softcap_logits(x, cap) == cap * tanh(x / cap)."""
    cap = 15.0
    logits = mx.array([0.0, 5.0, 100.0, -100.0])
    capped = softcap_logits(logits, cap)
    expected = cap * mx.tanh(logits / cap)
    assert mx.allclose(capped, expected)
    # A very large logit saturates near +cap but never reaches/exceeds it.
    big = float(capped[2].item())
    assert big < cap
    assert math.isclose(big, cap, abs_tol=1e-3)


def test_softcap_disabled_is_identity() -> None:
    """cap <= 0 leaves logits untouched."""
    logits = mx.array([1.0, -2.0, 3.5])
    assert mx.array_equal(softcap_logits(logits, 0.0), logits)
    assert mx.array_equal(softcap_logits(logits, -1.0), logits)


def test_softcap_changes_sampling_distribution() -> None:
    """Softcap compresses logit gaps, so it observably shifts the sampled mix.

    The greedy path can't reveal softcap (argmax is invariant under a monotone
    transform), so exercise it through the stochastic path where it alters the
    probability mass each token receives.
    """
    # One dominant token plus a long flat tail. Without softcap the dominant
    # token wins almost every draw; softcap (cap=2) shrinks the gap so the tail
    # is sampled far more often.
    logits = mx.concatenate([mx.array([20.0]), mx.zeros(31)])[None]

    def dominant_share(softcap: float, n: int = 3000) -> float:
        sampler = make_codebook_sampler(temperature=1.0, softcap=softcap)
        hits = 0
        for seed in range(n):
            mx.random.seed(seed)
            if int(sampler(logits)[0]) == 0:
                hits += 1
        return hits / n

    share_no_cap = dominant_share(0.0)
    share_cap = dominant_share(2.0)
    assert share_no_cap > 0.95  # uncapped: token 0 dominates
    assert share_cap < share_no_cap - 0.2  # softcap spreads mass to the tail


def test_top_p_matches_upstream_nucleus() -> None:
    """top-p must filter the temperature-scaled softmax mass like upstream.

    Upstream ``sample_tts`` applies top-p to ``softmax(logits / temp)``. With a
    cutoff that admits only the two leading tokens, no other id may be sampled.
    """
    logits = mx.array([[8.0, 7.5, 7.0, 1.0, -5.0]])
    probs = mx.softmax(logits, axis=-1)
    # cumulative mass of the top-2 tokens is ~0.813; a cutoff of 0.6 keeps
    # exactly tokens {0, 1} under upstream's "cumsum - prob > p" rule.
    sampler = make_codebook_sampler(temperature=1.0, top_p=0.6)
    sampled = set()
    for seed in range(2000):
        mx.random.seed(seed)
        sampled.add(int(sampler(logits)[0]))
    assert sampled <= {0, 1}, f"top-p leaked outside the nucleus: {sampled}"
    # sanity: the nucleus really is the two highest-probability tokens
    top2 = set(int(i) for i in mx.argsort(-probs[0], axis=-1)[:2].tolist())
    assert top2 == {0, 1}


def test_top_k_larger_than_vocab_does_not_crash() -> None:
    """An out-of-range top_k disables the filter instead of raising."""
    mx.random.seed(5)
    logits = mx.random.normal(shape=(2, 32))
    sampler = make_codebook_sampler(temperature=1.0, top_k=999)
    out = sampler(logits)  # must not raise
    assert out.shape == (2,)
    assert int(mx.max(out).item()) < 32


def test_output_shape_for_batched_codebooks() -> None:
    """[B, n_codebooks, vocab] -> [B, n_codebooks] for both greedy and sampling."""
    mx.random.seed(3)
    logits = mx.random.normal(shape=(2, 9, 1026))

    greedy = make_codebook_sampler(temperature=0.0)
    assert greedy(logits).shape == (2, 9)

    stochastic = make_codebook_sampler(temperature=1.0, top_p=0.9, top_k=50)
    out = stochastic(logits)
    assert out.shape == (2, 9)
    assert out.dtype == mx.int32
    # All sampled ids must lie within the vocab range.
    assert int(mx.min(out).item()) >= 0
    assert int(mx.max(out).item()) < 1026


def test_two_dim_vocab_only_shape() -> None:
    """A bare [..., vocab] input drops just the vocab axis."""
    mx.random.seed(4)
    logits = mx.random.normal(shape=(7, 1026))
    sampler = make_codebook_sampler(temperature=0.0)
    assert sampler(logits).shape == (7,)


# ── repetition penalty ────────────────────────────────────────────────────────


def test_repetition_penalty_divides_positive_logit_in_same_codebook() -> None:
    """A token seen in a codebook's window gets that codebook's logit divided."""
    # [B=1, C=2, V=4]: identical logits in both codebooks. Token 2 appears in
    # codebook 0's window only, so only codebook 0's logit for token 2 changes.
    logits = mx.array([[[1.0, 2.0, 4.0, 1.0], [1.0, 2.0, 4.0, 1.0]]])  # [1, 2, 4]
    # window: codebook 0 saw token 2; codebook 1 saw nothing (-1 padding).
    rep_ids = mx.array([[[2, 2], [-1, -1]]])  # [1, C=2, window=2]
    out = apply_repetition_penalty(logits, rep_ids, 2.0)
    out_np = np.asarray(out)
    # codebook 0: token 2 had +4.0 -> /2 = 2.0; others unchanged.
    assert np.isclose(out_np[0, 0, 2], 2.0)
    assert np.isclose(out_np[0, 0, 0], 1.0)
    # codebook 1: untouched (no repeats).
    assert np.allclose(out_np[0, 1], [1.0, 2.0, 4.0, 1.0])


def test_repetition_penalty_multiplies_negative_logit() -> None:
    """Negative logits are multiplied by the penalty (pushed further down)."""
    logits = mx.array([[[-3.0, 1.0, 2.0]]])  # [1, C=1, V=3]
    rep_ids = mx.array([[[0]]])  # token 0 repeated
    out = np.asarray(apply_repetition_penalty(logits, rep_ids, 1.5))
    assert np.isclose(out[0, 0, 0], -4.5)  # -3 * 1.5
    assert np.allclose(out[0, 0, 1:], [1.0, 2.0])


def test_repetition_penalty_ignores_codebooks_beyond_limit() -> None:
    """Only the first ``repetition_codebooks`` codebooks get penalized."""
    logits = mx.array([[[4.0, 1.0], [4.0, 1.0], [4.0, 1.0]]])  # [1, C=3, V=2]
    rep_ids = mx.array([[[0], [0], [0]]])  # token 0 repeated in every codebook
    out = np.asarray(
        apply_repetition_penalty(logits, rep_ids, 2.0, repetition_codebooks=2)
    )
    assert np.isclose(out[0, 0, 0], 2.0)  # codebook 0 penalized
    assert np.isclose(out[0, 1, 0], 2.0)  # codebook 1 penalized
    assert np.isclose(out[0, 2, 0], 4.0)  # codebook 2 (>= limit) untouched


def test_repetition_penalty_skips_ids_at_or_above_codebook_size() -> None:
    """eoa/pad ids (>= codebook_size) are not treated as repeats."""
    logits = mx.array([[[4.0, 1.0, 1.0]]])  # [1, C=1, V=3]
    # token 2 is "eoa/pad" when codebook_size=2; it must not penalize.
    rep_ids = mx.array([[[2]]])
    out = np.asarray(apply_repetition_penalty(logits, rep_ids, 2.0, codebook_size=2))
    assert np.allclose(out[0, 0], [4.0, 1.0, 1.0])  # unchanged


def test_repetition_penalty_disabled_passthrough() -> None:
    """penalty <= 1.0 or no window is a no-op."""
    logits = mx.array([[[4.0, 1.0]]])
    rep_ids = mx.array([[[0]]])
    assert mx.array_equal(apply_repetition_penalty(logits, rep_ids, 1.0), logits)
    assert mx.array_equal(apply_repetition_penalty(logits, None, 2.0), logits)


def test_repetition_penalty_shifts_sampling_away_from_repeated_token() -> None:
    """End-to-end: a repeated token is sampled less often once penalized."""
    # Token 0 dominates; without penalty it wins almost always.
    logits = mx.array([[[6.0, 0.0, 0.0, 0.0]]])  # [1, C=1, V=4]
    rep_ids = mx.array([[[0, 0, 0]]])  # token 0 heavily repeated
    sampler = make_codebook_sampler(
        temperature=1.0, repetition_penalty=2.0, codebook_size=4
    )

    def share(rep, n=2000):
        hits = 0
        for seed in range(n):
            mx.random.seed(seed)
            if int(sampler(logits, rep)[0, 0]) == 0:
                hits += 1
        return hits / n

    no_pen = share(None)
    pen = share(rep_ids)
    assert no_pen > 0.9
    assert pen < no_pen - 0.1  # penalty diverts mass off the repeated token


# ── min-p ─────────────────────────────────────────────────────────────────────


def test_apply_min_p_masks_low_probability_tokens() -> None:
    """probs below ``min_p * max_prob`` are zeroed and the rest renormalized."""
    probs = mx.array([[0.6, 0.3, 0.08, 0.02]])
    # max=0.6, min_p=0.2 -> threshold 0.12; keep {0.6, 0.3}, drop {0.08, 0.02}.
    out = np.asarray(apply_min_p(probs, 0.2))
    assert out[0, 2] == 0.0 and out[0, 3] == 0.0
    assert np.isclose(out[0].sum(), 1.0)
    assert np.isclose(out[0, 0], 0.6 / 0.9)
    assert np.isclose(out[0, 1], 0.3 / 0.9)


def test_min_p_disabled_passthrough() -> None:
    probs = mx.array([[0.5, 0.3, 0.2]])
    assert mx.array_equal(apply_min_p(probs, 0.0), probs)


def test_min_p_restricts_sampled_support() -> None:
    """A high min_p collapses sampling onto the dominant tokens only."""
    logits = mx.array([[8.0, 7.5, 2.0, -3.0]])  # [B=1, V=4]
    # softmax ~ [0.60, 0.36, 0.012, ...]; min_p=0.18 -> threshold ~0.108, keeps
    # only the two leading tokens.
    sampler = make_codebook_sampler(temperature=1.0, min_p=0.18)
    sampled = set()
    for seed in range(2000):
        mx.random.seed(seed)
        sampled.add(int(sampler(logits)[0]))
    assert sampled <= {0, 1}, f"min_p leaked low-probability tokens: {sampled}"
