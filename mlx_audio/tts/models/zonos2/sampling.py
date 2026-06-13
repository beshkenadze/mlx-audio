"""Per-codebook sampling for the ZONOS2 MLX port.

Mirrors the sampling semantics of upstream ``python/zonos2/tts/sampler.py``
(``sample_tts``): optional logit softcap, then temperature scaling, top-k and
top-p filtering, with greedy decoding when ``temperature <= 0``. Each audio
codebook is sampled independently along the last (vocab) axis, so a single
sampler handles both ``[..., n_codebooks, vocab]`` and ``[..., vocab]`` logits.

Upstream applies top-p to the *softmax probabilities* of the temperature-scaled
logits. ``mlx_lm.sample_utils.make_sampler`` expects **log-probabilities** as its
input (its ``apply_top_p`` exponentiates the input and thresholds on cumulative
mass), so we feed it ``log_softmax(logits / temperature)`` and let it own the
top-k / top-p / categorical chain. This reproduces the upstream nucleus cutoff
exactly while reusing the shared filtering kernels.

Reference: https://github.com/Zyphra/ZONOS2 -> python/zonos2/tts/sampler.py
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import make_sampler


def softcap_logits(logits: mx.array, cap: float) -> mx.array:
    """Squash logits into ``(-cap, cap)`` via ``cap * tanh(logits / cap)``.

    Mirrors the ZONOS2 ``loss_softcap`` applied to the output head before
    sampling. A non-positive ``cap`` disables the transform (identity).
    """
    if cap <= 0.0:
        return logits
    return cap * mx.tanh(logits / cap)


def make_codebook_sampler(
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
    softcap: float = 0.0,
) -> Callable[[mx.array], mx.array]:
    """Build a per-codebook sampler matching upstream ``sample_tts`` semantics.

    The returned ``sampler(logits)`` accepts logits shaped ``[..., vocab]`` or
    ``[..., n_codebooks, vocab]`` and returns ``int32`` token ids with the
    trailing vocab axis removed (one token per codebook / batch element).

    Args:
        temperature: Softmax temperature. ``<= 0`` selects greedy (``argmax``).
        top_k: Keep only the ``top_k`` highest-probability tokens (``0`` = off).
            Values ``>= vocab`` disable the filter (matching upstream).
        top_p: Nucleus cutoff in ``(0, 1)``; ``1.0`` (or ``0.0``) disables it.
        softcap: Logit softcap ``cap``; applied first when ``> 0``.

    Returns:
        A callable ``mx.array -> mx.array`` mapping logits to sampled token ids.
    """
    top_k = max(int(top_k), 0)

    if temperature <= 0:
        # Greedy: argmax over vocab, independent per codebook/batch element.
        # softcap is monotonic so it does not change the argmax, but apply it
        # anyway to keep the two branches behaviourally identical.
        def greedy_sampler(logits: mx.array) -> mx.array:
            logits = softcap_logits(logits, softcap)
            return mx.argmax(logits, axis=-1).astype(mx.int32)

        return greedy_sampler

    def sampler(logits: mx.array) -> mx.array:
        logits = softcap_logits(logits, softcap)
        # mlx_lm's apply_top_k raises for top_k >= vocab; upstream instead
        # treats an out-of-range top_k as "disabled", so clamp it here where the
        # vocab size is known.
        vocab = logits.shape[-1]
        effective_top_k = top_k if 0 < top_k < vocab else 0
        # mlx_lm's make_sampler expects log-probabilities; feeding it
        # log_softmax(logits / temp) makes top-p operate on the temperature
        # scaled probability mass exactly as upstream sample_tts does.
        logprobs = nn.log_softmax(logits / temperature, axis=-1)
        base_sampler = make_sampler(temp=1.0, top_p=top_p, top_k=effective_top_k)
        return base_sampler(logprobs).astype(mx.int32)

    return sampler
