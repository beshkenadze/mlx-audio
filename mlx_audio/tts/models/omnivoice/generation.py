import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .omnivoice import Model

AUDIO_MASK_ID = 1024


def cumulative_unmask_ratio(n: int, N: int = 32, tau: float = 0.1) -> float:
    """Cosine-like cumulative unmasking schedule from OmniVoice paper.

    r_n = tau * (n/N) / (1 + (tau-1) * (n/N))
    At n=0: r=0; at n=N: r=1 (regardless of tau).
    """
    if n == 0:
        return 0.0
    if n >= N:
        return 1.0
    t = n / N
    return tau * t / (1.0 + (tau - 1.0) * t)


def _gumbel_argmax(logits: mx.array, temperature: float = 5.0) -> mx.array:
    """Sample via Gumbel-max trick: argmax(logits/temp + Gumbel noise)."""
    gumbel = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape) + 1e-20) + 1e-20)
    return mx.argmax(logits / temperature + gumbel, axis=-1)


def _unmask_step(
    tokens: mx.array,
    frozen: mx.array,
    new_tokens: mx.array,
    confidence: mx.array,
    n_reveal: int,
) -> tuple:
    """Reveal n_reveal positions with highest confidence among non-frozen positions.

    Returns updated (tokens, frozen).
    """
    T, C = tokens.shape
    total = T * C

    # Zero out confidence for already-frozen positions
    masked_conf = mx.where(frozen, mx.array(-1.0), confidence)  # [T, C]
    flat_conf = masked_conf.reshape(-1)  # [T*C]

    # Double-argsort trick: rank[i] = position of element i in sorted order.
    # rank < n_reveal selects the top-n_reveal positions without allocating
    # the O(total * n_reveal) intermediate array that broadcasting would require.
    rank = mx.argsort(mx.argsort(-flat_conf))  # [T*C]
    reveal_mask_flat = rank < n_reveal
    reveal_mask = reveal_mask_flat.reshape(T, C)

    # Only update positions that are newly revealed (not already frozen)
    update_mask = reveal_mask & ~frozen
    tokens = mx.where(update_mask, new_tokens, tokens)
    frozen = frozen | update_mask
    return tokens, frozen


def iterative_unmask(
    model: "Model",
    cond_embeds: mx.array,  # [1, S, D]
    uncond_embeds: mx.array,  # [1, S, D]
    T: int,
    num_steps: int = 32,
    guidance_scale: float = 2.0,
    temperature: float = 5.0,
    tau: float = 0.1,
) -> mx.array:  # [T, 8] in [0, 1023]
    """Run iterative unmasking decode for OmniVoice audio token generation.

    Accepts pre-embedded conditioning (cond_embeds/uncond_embeds) so that
    voice cloning prefix embeddings can be injected once before the loop.
    """
    C = model.config.num_audio_codebook
    tokens = mx.full((T, C), AUDIO_MASK_ID, dtype=mx.int32)
    frozen = mx.zeros((T, C), dtype=mx.bool_)

    # Static prefix batch: [2, S, D] — computed once before the loop
    prefix_batch = mx.concatenate([cond_embeds, uncond_embeds], axis=0)
    prefix_len = cond_embeds.shape[1]

    for step in range(num_steps):
        tokens_batch = mx.stack([tokens, tokens], axis=0)  # [2, T, C]

        # Compute audio embeddings from current tokens
        audio_embeds = sum(
            model.audio_embeddings[i](tokens_batch[:, :, i]) for i in range(C)
        )  # [2, T, D]

        inputs_embeds = mx.concatenate(
            [prefix_batch, audio_embeds], axis=1
        )  # [2, S+T, D]

        logits_batch = model(inputs_embeds, prefix_len=prefix_len)  # [2, T, C, V]
        logits_cond = logits_batch[0:1]  # [1, T, C, V]
        logits_uncond = logits_batch[1:2]  # [1, T, C, V]

        # Classifier-free guidance
        logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        logits = logits[0]  # [T, C, V]

        # Sample new tokens and compute confidence
        new_tokens = _gumbel_argmax(logits, temperature)  # [T, C]
        confidence = mx.max(nn.softmax(logits, axis=-1), axis=-1)  # [T, C]

        # Number of total positions to have revealed after this step
        r_n = cumulative_unmask_ratio(step + 1, N=num_steps, tau=tau)
        n_reveal = max(1, math.floor(r_n * T * C))

        tokens, frozen = _unmask_step(tokens, frozen, new_tokens, confidence, n_reveal)

        # Materialize to prevent unbounded computation graph growth
        mx.eval(tokens, frozen)

    # Safety: replace any remaining mask tokens with token 0
    tokens = mx.where(tokens == AUDIO_MASK_ID, mx.zeros_like(tokens), tokens)
    return tokens
