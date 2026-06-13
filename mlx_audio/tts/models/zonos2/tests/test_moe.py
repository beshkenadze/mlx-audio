"""Synthetic shape/behaviour tests for the ZONOS2 MoE block (no GPU/CUDA).

Tiny config: dim=64, moe_n_experts=4, moe_router_dim=16, moe_start_from_layer=0.
The MoE intermediate is overridden via ``ffn_dim_multiplier``/``multiple_of`` so
the synthetic experts stay small.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.zonos2.config import ZONOS2Config
from mlx_audio.tts.models.zonos2.moe import (
    MoEFeedForward,
    Router,
    RouterMLP,
    split_sonic_w13,
)


def _realize(*arrays: mx.array) -> None:
    """Force MLX lazy arrays to materialize (sidesteps eval() naming)."""
    for arr in arrays:
        np.asarray(arr)


def _tiny_config(**overrides) -> ZONOS2Config:
    base = dict(
        n_layers=4,
        dim=64,
        head_dim=16,
        n_kv_heads=2,
        ffn_dim_multiplier=2.0,  # -> intermediate_size 128 (rounded to multiple_of)
        multiple_of=32,
        moe_n_experts=4,
        moe_router_dim=16,
        moe_router_topk=1,
        moe_start_from_layer=0,
        moe_end_from_layer=0,
    )
    base.update(overrides)
    return ZONOS2Config(**base)


def test_moe_feed_forward_preserves_shape():
    config = _tiny_config()
    moe = MoEFeedForward(config, layer_id=1)
    x = mx.random.normal((1, 3, 64))
    out = moe(x)
    assert out.shape == (1, 3, 64)
    assert out.dtype == x.dtype
    _realize(out)


def test_router_shapes_top1():
    config = _tiny_config()
    router = Router(config, layer_id=1)
    k = config.num_experts_per_tok(1)
    assert k == 1
    x = mx.random.normal((1, 3, 64))
    indices, weights = router(x)
    assert indices.shape == (1, 3, k)
    assert weights.shape == (1, 3, k)
    # Selected expert ids are valid.
    _realize(indices, weights)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < config.moe_n_experts


def test_router_top2_layer():
    # special_topk_layers makes layer 2 select two experts.
    config = _tiny_config(special_topk_layers={2: 2})
    assert config.num_experts_per_tok(2) == 2
    router = Router(config, layer_id=2)
    x = mx.random.normal((1, 5, 64))
    indices, weights = router(x)
    assert indices.shape == (1, 5, 2)
    assert weights.shape == (1, 5, 2)
    # Each token's two selected experts are distinct.
    pairs = np.asarray(indices)
    for row in pairs[0]:
        assert row[0] != row[1]


def test_router_weights_are_unbiased_softmax_probs():
    # Returned weights must be the original (un-biased) softmax probabilities,
    # so each lies in (0, 1); with norm_topk_prob False they are NOT renormalized.
    config = _tiny_config(special_topk_layers={2: 2})
    router = Router(config, layer_id=2)
    x = mx.random.normal((1, 4, 64))
    _, weights = router(x)
    w = np.asarray(weights)[0]
    for row in w:
        for p in row:
            assert 0.0 < p < 1.0
        # top-2 probabilities are NOT renormalized (no forced sum to 1).
        assert float(sum(row)) < 1.0 + 1e-4


def test_router_balancing_bias_steers_selection_only():
    # A large bias on one expert should force its selection while the returned
    # weight stays the original softmax prob (not the biased score).
    config = _tiny_config()
    router = Router(config, layer_id=1)
    forced = 2
    bias = [0.0] * config.moe_n_experts
    bias[forced] = 100.0
    router.balancing_biases = mx.array(bias, dtype=mx.float32)
    x = mx.random.normal((1, 6, 64))
    indices, weights = router(x)
    idx = np.asarray(indices).reshape(-1)
    wt = np.asarray(weights).reshape(-1)
    assert all(int(i) == forced for i in idx)
    # Weight is a plain softmax prob (< 1), not the +100 biased score.
    assert all(0.0 < float(p) < 1.0 for p in wt)


def test_eda_blend_changes_router_states():
    # With EDA active (layer_id != moe_start_from_layer) passing router_states
    # should alter the routing relative to no router_states.
    config = _tiny_config()
    router = Router(config, layer_id=1)
    assert router.use_eda
    x = mx.random.normal((1, 3, 64))
    rs = mx.random.normal((1, 3, config.moe_router_dim)) * 10.0
    _, _, next_a = router(x, return_router_states=True)
    _, _, next_b = router(x, rs, return_router_states=True)
    assert next_a.shape == (1, 3, config.moe_router_dim)
    # router_states_next differs once EDA blending is applied.
    assert not bool(mx.allclose(next_a, next_b))


def test_first_moe_layer_has_no_eda():
    config = _tiny_config()
    router = Router(config, layer_id=config.moe_start_from_layer)
    assert not router.use_eda
    assert "router_states_scale" not in dict(router.parameters())


def test_routermlp_structure_and_shape():
    mlp = RouterMLP(router_dim=16, num_experts=4)
    x = mx.random.normal((2, 16))
    out = mlp(x)
    assert out.shape == (2, 4)
    # Mirror upstream nn.Sequential numbering: linears at 0/2/4, GELU at 1/3.
    assert mlp.layers[1].__class__.__name__ == "GELU"
    assert mlp.layers[3].__class__.__name__ == "GELU"


def test_split_sonic_w13_deinterleaves():
    experts, inter, hidden = 2, 3, 5
    # Build w13 so even rows are "gate" and odd rows are "up".
    gate_src = mx.arange(experts * inter * hidden).reshape(experts, inter, hidden)
    up_src = gate_src + 1000
    # Interleave: row 0=gate0, row 1=up0, row 2=gate1, ...
    rows = []
    for i in range(inter):
        rows.append(gate_src[:, i : i + 1, :])
        rows.append(up_src[:, i : i + 1, :])
    w13 = mx.concatenate(rows, axis=1)
    gate, up = split_sonic_w13(w13)
    assert gate.shape == (experts, inter, hidden)
    assert up.shape == (experts, inter, hidden)
    assert bool(mx.array_equal(gate, gate_src))
    assert bool(mx.array_equal(up, up_src))
