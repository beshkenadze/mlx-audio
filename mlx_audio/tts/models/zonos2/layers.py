# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""Core transformer layers for the ZONOS2 MLX port.

Mirrors the upstream PyTorch implementation
(``python/zonos2/models/zonos2.py`` classes ``Attention``, ``FeedForward``,
``softcap``; ``python/zonos2/layers/{rotary,norm}.py``).

Notable ZONOS2 specifics faithfully reproduced here:

* **GQA** with ``num_qo_heads=16`` query heads and ``n_kv_heads=4`` KV heads
  (``head_dim=128``).
* **Fused KV projection** ``wkv`` whose checkpoint weight is rank-3
  ``[2, kv_heads*head_dim, hidden]`` (``[0]`` = K, ``[1]`` = V).
* **Per-head QK RMSNorm** (eps ``1e-6``, no affine weight) followed by a
  learnable temperature ``temp`` (shape ``[1, num_heads, 1]``) applied to the
  query only via ``q *= temp.abs()``.
* **Headwise sigmoid gating** ``gate = sigmoid(gater(x))`` multiplied into the
  attention output per head before the output projection.
* **Interleaved RoPE** (upstream ``is_neox=False``).
* **Dense SwiGLU** with a single fused ``w_in`` weight of shape
  ``[2, intermediate, hidden]`` where ``[0]`` is the "up" path ``h`` and ``[1]``
  is the SiLU-gated path: ``y = w_out(h * silu(gate))``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.tts.models.zonos2.config import ZONOS2Config


class RMSNorm(nn.Module):
    """Root-Mean-Square layer normalization via ``mx.fast.rms_norm``."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def apply_rotary(
    q: mx.array,
    k: mx.array,
    *,
    offset: int = 0,
    base: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    """Apply *interleaved* RoPE to ``q``/``k`` of shape ``[B, T, H, D]``.

    Matches the upstream ZONOS2 ``is_neox=False`` convention: consecutive
    ``(even, odd)`` pairs of the head dimension form the complex plane that is
    rotated. ``offset`` shifts the absolute positions (KV-cache decoding).
    """
    t = q.shape[1]
    d = q.shape[-1]
    if d % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")
    half = d // 2

    freqs = mx.exp(mx.arange(half, dtype=mx.float32) * (-math.log(base) * 2.0 / d))
    ts = (mx.arange(t, dtype=mx.float32) + float(offset))[:, None]  # [T, 1]
    angles = ts * freqs[None, :]  # [T, half]
    cos = mx.cos(angles)[None, :, None, :]  # [1, T, 1, half]
    sin = mx.sin(angles)[None, :, None, :]

    def _rotate(x: mx.array) -> mx.array:
        # GQA: q and k may have different head counts, so derive shape per tensor.
        b, _, h, _ = x.shape
        x = x.reshape(b, t, h, half, 2)
        xr = x[..., 0].astype(mx.float32)
        xi = x[..., 1].astype(mx.float32)
        out = mx.stack([xr * cos - xi * sin, xr * sin + xi * cos], axis=-1)
        return out.astype(x.dtype).reshape(b, t, h, d)

    return _rotate(q), _rotate(k)


class RotaryEmbedding(nn.Module):
    """Interleaved RoPE for ``head_dim=128``, ``base=config.rope_theta``."""

    def __init__(self, head_dim: int = 128, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = float(base)

    def __call__(
        self, q: mx.array, k: mx.array, offset: int = 0
    ) -> tuple[mx.array, mx.array]:
        return apply_rotary(q, k, offset=offset, base=self.base)


class Attention(nn.Module):
    """GQA attention with QK-norm, learnable temperature, and headwise gating.

    Checkpoint naming (mirrors upstream ``Attention``):

    * ``attention.wq.weight``  -> ``[num_heads*head_dim, hidden]``
    * ``attention.wkv.weight`` -> rank-3 ``[2, kv_heads*head_dim, hidden]``
    * ``attention.wo.weight``  -> ``[hidden, num_heads*head_dim]``
    * ``attention.gater.weight`` -> ``[num_heads, hidden]``
    * ``attention.temp``       -> ``[1, num_heads, 1]``
    """

    def __init__(self, config: ZONOS2Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.dim

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.wq = nn.Linear(self.hidden_size, q_dim, bias=False)
        # Fused K/V projection: weight is rank-3 [2, kv_dim, hidden] in the
        # checkpoint ([0] = K, [1] = V). Stored as a raw parameter so the key
        # name `attention.wkv.weight` matches.
        self.wkv = _FusedProj(self.hidden_size, kv_dim)
        self.wo = nn.Linear(q_dim, self.hidden_size, bias=False)
        self.gater = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Learnable QK-norm temperature, applied to the query only.
        self.temp = mx.ones((1, self.num_heads, 1))

        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # Upstream QK RMSNorm uses eps 1e-6 with a unit (non-learnable) weight.
        self._qk_norm_eps = 1e-6
        self._qk_norm_weight = mx.ones((self.head_dim,))

    def _qk_rms_norm(self, x: mx.array) -> mx.array:
        # Per-head RMSNorm over `head_dim` (matches upstream input dtype).
        return mx.fast.rms_norm(
            x, self._qk_norm_weight.astype(x.dtype), self._qk_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        B, T, _ = x.shape

        # RoPE positions must start at the number of already-cached tokens.
        # Honour an explicit `offset`, but prefer the cache's own count so the
        # query positions can never desync from the stored keys.
        if cache is not None and offset == 0:
            offset = cache.offset

        # Headwise gate computed in fp32. The sigmoid is evaluated on the fp32
        # gater pre-activation (the CUDA reference computes the gating sigmoid in
        # fp32), so the per-head gate is not pre-quantized to bf16 before it scales
        # the attention context. A bf16-rounded gate perturbs the gated context
        # that feeds `wo`, nudging the residual onto bf16 argmax ties.
        gate = mx.sigmoid(self.gater(x).astype(mx.float32))  # [B, T, num_heads]

        q = self.wq(x).reshape(B, T, self.num_heads, self.head_dim)
        k, v = self.wkv(x)  # each [B, T, kv_dim]
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)

        # QK norm (input dtype, matching upstream) + temperature on the query.
        q = self._qk_rms_norm(q) * self.temp.abs().astype(q.dtype)
        k = self._qk_rms_norm(k)

        # Interleaved RoPE with absolute positions (cache offset aware).
        q, k = self.rotary(q, k, offset=offset)

        # [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # [B, T, H, D] then headwise gating before the output projection. The
        # gate is fp32; cast the gated context back to the activation dtype so the
        # `wo` projection stays a bf16 GEMM like upstream (only the gate itself is
        # promoted, not the output matmul).
        out_dtype = out.dtype
        out = out.transpose(0, 2, 1, 3)
        out = (out * gate[..., None]).astype(out_dtype)
        out = out.reshape(B, T, self.num_heads * self.head_dim)
        return self.wo(out)


class _FusedProj(nn.Module):
    """Two-way fused projection holding a single rank-3 weight ``[2, out, in]``.

    Reproduces upstream ``ChunkedLinear`` with ``divisor=2`` (used for both
    ``attention.wkv`` and ``feed_forward.w_in``). Storing the weight as one
    rank-3 parameter named ``weight`` keeps the checkpoint key (``*.weight``)
    intact while splitting the two output halves ``[0]`` and ``[1]``.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = mx.zeros((2, out_features, in_features))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # weight: [2, out, in] -> contraction over `in` for each half.
        return x @ self.weight[0].T, x @ self.weight[1].T


class FeedForward(nn.Module):
    """Dense SwiGLU with a fused ``w_in`` weight ``[2, intermediate, hidden]``.

    Checkpoint naming:

    * ``feed_forward.w_in.weight``  -> ``[2, intermediate, hidden]``
      (``[0]`` = up path ``h``, ``[1]`` = gate path through SiLU).
    * ``feed_forward.w_out.weight`` -> ``[hidden, intermediate]``.

    Output: ``w_out(h * silu(gate))``.
    """

    def __init__(self, config: ZONOS2Config):
        super().__init__()
        hidden = config.dim
        inter = config.intermediate_size
        # w_in.weight is rank-3 [2, inter, hidden]: [0] = up path `h`, [1] = gate.
        self.w_in = _FusedProj(hidden, inter)
        self.w_out = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h, gate = self.w_in(x)
        return self.w_out(h * nn.silu(gate))


def softcap(x: mx.array, cap: float) -> mx.array:
    """Soft-cap logits: ``cap * tanh(x / cap)``."""
    return cap * mx.tanh(x / cap)
