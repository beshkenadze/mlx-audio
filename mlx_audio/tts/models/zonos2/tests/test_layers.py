"""Synthetic unit tests for ZONOS2 core layers (Mac, no GPU).

Shapes only — full CUDA parity is the coordinator Phase-D gate.
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from mlx_audio.tts.models.zonos2.config import ZONOS2Config
from mlx_audio.tts.models.zonos2.layers import (
    Attention,
    FeedForward,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary,
    softcap,
)


def _tiny_config() -> ZONOS2Config:
    return ZONOS2Config(
        n_layers=4,
        dim=64,
        head_dim=16,
        n_heads=4,  # dim // head_dim
        n_kv_heads=2,
        moe_n_experts=1,
        rope_theta=10000.0,
    )


def test_rmsnorm_shape_and_value():
    dims = 64
    norm = RMSNorm(dims, eps=1e-5)
    x = mx.random.normal((2, 5, dims))
    y = norm(x)
    assert y.shape == x.shape

    # Manual reference rms_norm with unit weight.
    ms = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    ref = (x.astype(mx.float32) * mx.rsqrt(ms + 1e-5)).astype(x.dtype)
    assert mx.allclose(y, ref, atol=1e-4)


def test_softcap_matches_formula():
    x = mx.random.normal((3, 7)) * 30.0
    cap = 15.0
    assert mx.allclose(softcap(x, cap), cap * mx.tanh(x / cap), atol=1e-6)


def test_apply_rotary_is_interleaved_and_preserves_shape():
    b, t, h, d = 1, 5, 2, 16
    q = mx.random.normal((b, t, h, d))
    k = mx.random.normal((b, t, h, d))
    qo, ko = apply_rotary(q, k, offset=0, base=10000.0)
    assert qo.shape == q.shape and ko.shape == k.shape

    # Position 0 is the identity rotation (angles are all zero).
    assert mx.allclose(qo[:, 0], q[:, 0], atol=1e-4)
    assert mx.allclose(ko[:, 0], k[:, 0], atol=1e-4)

    # Interleaved rotation of the first (even, odd) pair at position 1.
    half = d // 2
    freqs = mx.exp(mx.arange(half, dtype=mx.float32) * (-math.log(10000.0) * 2.0 / d))
    ang = float(freqs[0])  # position 1 -> angle = 1 * freqs[0]
    x0 = float(q[0, 1, 0, 0])
    x1 = float(q[0, 1, 0, 1])
    exp0 = x0 * math.cos(ang) - x1 * math.sin(ang)
    exp1 = x0 * math.sin(ang) + x1 * math.cos(ang)
    assert abs(float(qo[0, 1, 0, 0]) - exp0) < 1e-3
    assert abs(float(qo[0, 1, 0, 1]) - exp1) < 1e-3


def test_rotary_embedding_offset_shifts_positions():
    rope = RotaryEmbedding(head_dim=16, base=10000.0)
    q = mx.random.normal((1, 3, 2, 16))
    k = mx.random.normal((1, 3, 2, 16))
    # Token at position p with offset=2 equals token at absolute position p+2.
    q_off, _ = rope(q, k, offset=2)
    q_full, _ = apply_rotary(
        mx.concatenate([mx.zeros((1, 2, 2, 16)), q], axis=1),
        mx.concatenate([mx.zeros((1, 2, 2, 16)), k], axis=1),
        offset=0,
    )
    assert mx.allclose(q_off, q_full[:, 2:], atol=1e-3)


def test_attention_shape():
    cfg = _tiny_config()
    attn = Attention(cfg, layer_id=0)
    x = mx.random.normal((1, 5, cfg.dim))
    causal = nn.MultiHeadAttention.create_additive_causal_mask(5)
    y = attn(x, mask=causal)
    assert y.shape == (1, 5, cfg.dim)


def _randomize_attention(attn: Attention) -> None:
    # The fused/linear weights default to zeros; give them real values so the
    # output is not degenerately ~0 (which would mask any behavioral check).
    attn.wq.weight = mx.random.normal(attn.wq.weight.shape) * 0.3
    attn.wkv.weight = mx.random.normal(attn.wkv.weight.shape) * 0.3
    attn.wo.weight = mx.random.normal(attn.wo.weight.shape) * 0.3
    attn.gater.weight = mx.random.normal(attn.gater.weight.shape) * 0.3


def test_attention_causal_mask_changes_output():
    cfg = _tiny_config()
    attn = Attention(cfg, layer_id=0)
    _randomize_attention(attn)
    x = mx.random.normal((1, 5, cfg.dim))
    causal = nn.MultiHeadAttention.create_additive_causal_mask(5)
    masked = attn(x, mask=causal)
    full = attn(x, mask=None)
    # Causal masking must actually restrict attention -> different output.
    assert not bool(mx.allclose(masked, full, atol=1e-4))


def test_fused_kv_split_ordering():
    cfg = _tiny_config()
    attn = Attention(cfg, layer_id=0)
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    k_w = mx.random.normal((kv_dim, cfg.dim)) * 0.05
    v_w = mx.random.normal((kv_dim, cfg.dim)) * 0.05
    # weight[0] -> K, weight[1] -> V (mirrors upstream ChunkedLinear split).
    attn.wkv.weight = mx.stack([k_w, v_w], axis=0)
    x = mx.random.normal((1, 4, cfg.dim))
    k, v = attn.wkv(x)
    assert mx.allclose(k, x @ k_w.T, atol=1e-4)
    assert mx.allclose(v, x @ v_w.T, atol=1e-4)


def test_attention_incremental_matches_prefill():
    cfg = _tiny_config()
    attn = Attention(cfg, layer_id=0)
    _randomize_attention(attn)
    x = mx.random.normal((1, 4, cfg.dim))

    # Offline: full sequence at once with a causal mask.
    causal = nn.MultiHeadAttention.create_additive_causal_mask(4)
    offline = attn(x, mask=causal)

    # Incremental: feed one token at a time through a KV cache.
    cache = KVCache()
    steps = []
    for t in range(4):
        step = attn(x[:, t : t + 1], mask=None, cache=cache, offset=t)
        steps.append(step)
    incremental = mx.concatenate(steps, axis=1)

    assert incremental.shape == offline.shape
    assert mx.allclose(incremental, offline, atol=1e-3)


def test_attention_cache_derives_offset():
    cfg = _tiny_config()
    attn = Attention(cfg, layer_id=0)
    _randomize_attention(attn)
    x = mx.random.normal((1, 4, cfg.dim))
    causal = nn.MultiHeadAttention.create_additive_causal_mask(4)
    offline = attn(x, mask=causal)

    # Same as above but rely on cache.offset (no explicit offset passed).
    cache = KVCache()
    steps = [attn(x[:, t : t + 1], mask=None, cache=cache) for t in range(4)]
    incremental = mx.concatenate(steps, axis=1)
    assert mx.allclose(incremental, offline, atol=1e-3)


def test_attention_projection_shapes():
    cfg = _tiny_config()
    attn = Attention(cfg, layer_id=1)
    assert attn.wq.weight.shape == (cfg.num_qo_heads * cfg.head_dim, cfg.dim)
    # Fused KV: rank-3 [2, kv_heads*head_dim, hidden].
    assert attn.wkv.weight.shape == (2, cfg.n_kv_heads * cfg.head_dim, cfg.dim)
    assert attn.wo.weight.shape == (cfg.dim, cfg.num_qo_heads * cfg.head_dim)
    assert attn.gater.weight.shape == (cfg.num_qo_heads, cfg.dim)
    assert attn.temp.shape == (1, cfg.num_qo_heads, 1)


def test_feedforward_shape_and_fused_w_in():
    cfg = _tiny_config()
    ff = FeedForward(cfg)
    inter = cfg.intermediate_size

    # Fused w_in weight is rank-3 [2, intermediate, hidden].
    assert ff.w_in.weight.shape == (2, inter, cfg.dim)
    assert ff.w_out.weight.shape == (cfg.dim, inter)

    # Hand-build a fused w_in and verify y == w_out(h * silu(gate)).
    up_w = mx.random.normal((inter, cfg.dim)) * 0.05
    gate_w = mx.random.normal((inter, cfg.dim)) * 0.05
    out_w = mx.random.normal((cfg.dim, inter)) * 0.05
    ff.w_in.weight = mx.stack([up_w, gate_w], axis=0)
    ff.w_out.weight = out_w

    x = mx.random.normal((1, 5, cfg.dim))
    y = ff(x)
    assert y.shape == (1, 5, cfg.dim)

    h = x @ up_w.T
    gate = x @ gate_w.T
    ref = (h * nn.silu(gate)) @ out_w.T
    assert mx.allclose(y, ref, atol=1e-4)
