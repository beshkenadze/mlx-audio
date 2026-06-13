"""Synthetic tests for the ZONOS2 per-codebook sampler.

No GPU / CUDA fixtures: pure shape + semantics checks on random logits.
"""

from __future__ import annotations

import math

import mlx.core as mx

from mlx_audio.tts.models.zonos2.sampling import make_codebook_sampler, softcap_logits


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
