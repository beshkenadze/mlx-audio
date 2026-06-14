"""ZONOS2 sparse-MoE block (MLX port).

Mirrors the upstream ``zonos2.models.zonos2`` MoE stack:

* ``Router``            -> ``Router`` + ``RouterMLP`` (EDA + balancing-bias top-k)
* ``MoEFeedForward``    -> ``MoEFeedForward`` + ``FusedGroupedExperts``

The fused ``w13`` (interleaved gate/up) -> split conversion lives in
``convert.py``; here the experts already consume SEPARATE ``gate_proj`` /
``up_proj`` / ``down_proj`` weights via :class:`mlx_lm.models.switch_layers.SwitchGLU`
(``mx.gather_mm``).

Reference: https://github.com/Zyphra/ZONOS2 -> python/zonos2/models/zonos2.py
(``Router`` ~L523-626, ``RouterMLP`` ~L457-509, ``FusedGroupedExperts`` ~L334-454,
``MoEFeedForward`` ~L629-671).

Parity notes (matching the released checkpoint):
* Router math runs in fp32 (``balancing_biases``, softmax, top-k selection).
* ``norm_topk_prob`` is ``False`` upstream, so the top-2 layer (26) does NOT
  renormalize the selected weights -- it keeps the raw softmax probabilities.
* The balancing bias steers expert SELECTION only; the returned weights are the
  ORIGINAL (un-biased) softmax probabilities of the chosen experts.
* GELU is the exact (erf) form, matching ``torch.nn.functional.gelu`` default.
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.bailing_moe import aggregate_expert_outputs
from mlx_lm.models.switch_layers import SwitchGLU

from mlx_audio.tts.models.zonos2.config import ZONOS2Config


def split_sonic_w13(w13: mx.array) -> Tuple[mx.array, mx.array]:
    """De-interleave a SonicMoE fused ``w13`` into ``(gate, up)``.

    ``w13`` is rank-3 ``[experts, 2 * intermediate, hidden]`` with gate/up rows
    interleaved (gate = even rows, up = odd rows). Mirrors upstream
    ``_convert_sonic_w13_to_gate_up`` (which concatenates ``[gate, up]``); here we
    return the two halves separately for ``SwitchGLU``'s split projections.

    Used by ``convert.py`` -- kept here so the de-interleave convention has a
    single source of truth.
    """
    if w13.ndim != 3:
        raise ValueError(f"Expected SonicMoE w13 to be rank-3, got {w13.shape}")
    if w13.shape[1] % 2 != 0:
        raise ValueError(f"Expected even fused width in SonicMoE w13, got {w13.shape}")
    gate = w13[:, 0::2, :]
    up = w13[:, 1::2, :]
    return gate, up


class RouterMLP(nn.Module):
    """Router MLP mirroring upstream ``nn.Sequential`` numeric naming.

    Sequential structure (GELU at indices 1/3 are stateless):
        0: Linear(router_dim, router_dim, bias=True)
        1: GELU
        2: Linear(router_dim, router_dim, bias=True)
        3: GELU
        4: Linear(router_dim, num_experts, bias=False)
    """

    def __init__(self, router_dim: int, num_experts: int):
        super().__init__()
        # Exact (erf) GELU to match torch.nn.functional.gelu's default; MLX's
        # ``approx="precise"`` is the *tanh* approximation (~5e-4 error), so the
        # default ``nn.GELU()`` is the parity-correct choice here.
        self.layers = [
            nn.Linear(router_dim, router_dim, bias=True),
            nn.GELU(),
            nn.Linear(router_dim, router_dim, bias=True),
            nn.GELU(),
            nn.Linear(router_dim, num_experts, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class Router(nn.Module):
    """MoE router with EDA blending + balancing-bias-aware top-k selection.

    ``__call__(x) -> (indices[B, T, k], weights[B, T, k])`` where ``k`` is
    ``config.num_experts_per_tok(layer_id)``.

    Checkpoint keys (under ``...feed_forward.router.``):
        down_proj.{weight,bias}          Linear(hidden -> router_dim)
        rmsnorm_eda.weight               RMSNorm(router_dim)
        router_mlp.{0,2,4}.weight, .{0,2}.bias   GELU MLP -> num_experts logits
        balancing_biases [num_experts]   fp32, added before top-k
        router_states_scale [router_dim] EDA layers only
    """

    def __init__(self, config: ZONOS2Config, layer_id: int):
        super().__init__()
        self.hidden_size = config.dim
        self.router_dim = config.moe_router_dim
        self.num_experts = config.moe_n_experts
        self.top_k = config.num_experts_per_tok(layer_id)
        # ``norm_topk_prob`` is False for the released checkpoint -> never renorm.
        self.norm_topk_prob = False

        self.down_proj = nn.Linear(self.hidden_size, self.router_dim, bias=True)
        self.router_mlp = RouterMLP(self.router_dim, self.num_experts)
        self.rmsnorm_eda = nn.RMSNorm(self.router_dim, eps=config.norm_eps)

        # EDA is active for every MoE layer except the first MoE layer.
        self.use_eda = layer_id != config.moe_start_from_layer
        if self.use_eda:
            self.router_states_scale = mx.ones((self.router_dim,))

        # Balancing biases steer SELECTION only (fp32); legacy strategy adds them.
        self.balancing_biases = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        router_states: Optional[mx.array] = None,
        *,
        return_router_states: bool = False,
    ):
        """Route ``x`` to top-k experts.

        Args:
            x: ``[B, T, dim]`` hidden states.
            router_states: ``[B, T, router_dim]`` from the previous MoE layer
                (EDA chaining). When ``None`` the EDA blend is skipped, matching
                upstream's ``if self.use_eda and router_states is not None``.
            return_router_states: when True also return ``router_states_next``
                (the pre-norm router hidden states) for the next layer's EDA.

        Returns:
            ``(indices[B, T, k], weights[B, T, k])`` or, with
            ``return_router_states``, ``(indices, weights, router_states_next)``.
        """
        hidden = self.down_proj(x)

        if self.use_eda and router_states is not None:
            hidden = hidden + router_states * self.router_states_scale

        # Saved BEFORE normalization for the next layer's EDA blend.
        router_states_next = hidden

        hidden = self.rmsnorm_eda(hidden)

        # Router math is fp32 (parity with the checkpoint).
        logits = self.router_mlp(hidden).astype(mx.float32)
        expert_prob = mx.softmax(logits, axis=-1)

        # Legacy balancing: bias added to scores used only for top-k selection.
        routing_scores = expert_prob + self.balancing_biases

        k = self.top_k
        indices = mx.argpartition(routing_scores, kth=-k, axis=-1)[..., -k:]
        # Order selected experts by descending routing score (stable downstream).
        selected = mx.take_along_axis(routing_scores, indices, axis=-1)
        order = mx.argsort(-selected, axis=-1)
        indices = mx.take_along_axis(indices, order, axis=-1)

        # Returned weights are the ORIGINAL (un-biased) probabilities.
        weights = mx.take_along_axis(expert_prob, indices, axis=-1)
        if k > 1 and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)

        if return_router_states:
            return indices, weights, router_states_next
        return indices, weights


class MoEFeedForward(nn.Module):
    """Sparse MoE feed-forward: route via :class:`Router`, run :class:`SwitchGLU`.

    ``__call__(x) -> [B, T, dim]``. Experts use SEPARATE gate/up/down weights
    (the fused ``w13`` split is done in ``convert.py``).
    """

    def __init__(self, config: ZONOS2Config, layer_id: int):
        super().__init__()
        self.router = Router(config, layer_id)
        moe_inter = (
            config.moe_intermediate_size
            if config.moe_intermediate_size > 0
            else config.intermediate_size
        )
        self.experts = SwitchGLU(
            input_dims=config.dim,
            hidden_dims=moe_inter,
            num_experts=config.moe_n_experts,
            bias=False,
        )

    def __call__(
        self,
        x: mx.array,
        router_states: Optional[mx.array] = None,
        *,
        return_router_states: bool = False,
    ):
        """Apply the MoE block.

        Args mirror :meth:`Router.__call__`; with ``return_router_states`` the
        next-layer EDA states are returned alongside the output.
        """
        in_dtype = x.dtype
        indices, weights, router_states_next = self.router(
            x, router_states, return_router_states=True
        )

        expert_out = self.experts(x, indices)
        out = aggregate_expert_outputs(
            expert_out.astype(mx.float32), weights.astype(mx.float32)
        ).astype(in_dtype)

        if return_router_states:
            return out, router_states_next
        return out
