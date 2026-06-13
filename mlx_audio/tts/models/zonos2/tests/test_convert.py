"""Synthetic-fixture tests for the ZONOS2 torch->MLX weight converter.

No real 8B weights and no ``torch`` are required: a small in-memory MLX
state-dict exercises every remap branch (parametrization strip, SonicMoE w13
de-interleave, router pass-through, dtype policy). The full ``convert`` path is
exercised by monkeypatching the torch loader so a fixture dict flows through the
real save/reload + dtype-policy code.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.zonos2 import convert as convert_mod
from mlx_audio.tts.models.zonos2.convert import (
    SWITCH_MLP_PREFIX,
    _torch_state_dict_to_mx,
    convert,
    remap_state_dict,
    split_sonic_w13,
)

HIDDEN = 8
INTER = 4
EXPERTS = 3


def _interleaved_w13(gate: mx.array, up: mx.array) -> mx.array:
    """Build a fused w13 [E, 2*inter, hidden] with gate=even rows, up=odd rows."""
    e, inter, hidden = gate.shape
    fused = mx.zeros((e, 2 * inter, hidden))
    fused[:, 0::2, :] = gate
    fused[:, 1::2, :] = up
    return fused


def _make_state_dict() -> dict[str, mx.array]:
    """A synthetic ZONOS2 state-dict touching every remap branch."""
    rng = np.random.default_rng(0)

    def arr(*shape: int) -> mx.array:
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    gate = arr(EXPERTS, INTER, HIDDEN)
    up = arr(EXPERTS, INTER, HIDDEN)

    return {
        # embeddings / head -> pass through
        "multi_embedder.embedders.0.weight": arr(1026, HIDDEN),
        "multi_embedder.embedders.8.weight": arr(1026, HIDDEN),
        "multi_output.weight": arr(9754, HIDDEN),
        # dense FFN (fused w_in stays rank-3, untouched)
        "layers.0.feed_forward.w_in.weight": arr(2, INTER, HIDDEN),
        "layers.0.feed_forward.w_out.weight": arr(HIDDEN, INTER),
        # norms -> fp32
        "layers.0.attention_norm.weight": arr(HIDDEN),
        "layers.0.ffn_norm.weight": arr(HIDDEN),
        "norm.weight": arr(HIDDEN),
        # MoE layer: fused SonicMoE experts -> split into gate/up/down
        "layers.5.feed_forward.experts.w13": _interleaved_w13(gate, up),
        "layers.5.feed_forward.experts.w2": arr(EXPERTS, HIDDEN, INTER),
        # router -> pass through, fp32
        "layers.5.feed_forward.router.down_proj.weight": arr(128, HIDDEN),
        "layers.5.feed_forward.router.down_proj.bias": arr(128),
        "layers.5.feed_forward.router.rmsnorm_eda.weight": arr(128),
        "layers.5.feed_forward.router.router_mlp.0.weight": arr(128, 128),
        "layers.5.feed_forward.router.router_mlp.0.bias": arr(128),
        "layers.5.feed_forward.router.router_mlp.2.weight": arr(128, 128),
        "layers.5.feed_forward.router.router_mlp.2.bias": arr(128),
        "layers.5.feed_forward.router.router_mlp.4.weight": arr(16, 128),
        "layers.5.feed_forward.router.balancing_biases": arr(16),
        "layers.5.feed_forward.router.router_states_scale": arr(1),
        # training-only router stats -> dropped
        "layers.5.feed_forward.router.ent_denom": arr(1),
        "layers.5.feed_forward.router.normalized_entropy": arr(1),
        # weight-norm parametrization (modern original0/original1 pair)
        "wn.parametrizations.weight.original0": arr(HIDDEN, 1),
        "wn.parametrizations.weight.original1": arr(HIDDEN, HIDDEN),
        # weight-norm parametrization (single .original -> straight rename)
        "single.parametrizations.weight.original": arr(HIDDEN, HIDDEN),
    }, (gate, up)


# ── split_sonic_w13 ───────────────────────────────────────────────────────────


def test_split_sonic_w13_deinterleaves_even_gate_odd_up():
    gate = mx.arange(EXPERTS * INTER * HIDDEN).reshape(EXPERTS, INTER, HIDDEN)
    up = gate + 1000.0
    fused = _interleaved_w13(gate, up)

    got_gate, got_up = split_sonic_w13(fused)

    assert got_gate.shape == (EXPERTS, INTER, HIDDEN)
    assert got_up.shape == (EXPERTS, INTER, HIDDEN)
    assert mx.array_equal(got_gate, gate)
    assert mx.array_equal(got_up, up)


def test_split_sonic_w13_rejects_bad_rank_and_odd_width():
    import pytest

    with pytest.raises(ValueError):
        split_sonic_w13(mx.zeros((EXPERTS, 2 * INTER)))  # rank-2
    with pytest.raises(ValueError):
        split_sonic_w13(mx.zeros((EXPERTS, 2 * INTER + 1, HIDDEN)))  # odd width


# ── remap_state_dict ──────────────────────────────────────────────────────────


def test_remap_strips_weight_norm_parametrization():
    weights, _ = _make_state_dict()
    out = remap_state_dict(weights, mx.bfloat16)

    # parametrization keys gone, collapsed names present
    assert not any(".parametrizations." in k for k in out)
    assert "wn.weight" in out
    assert "single.weight" in out
    # single .original is a straight rename (values preserved up to dtype cast)
    expected = weights["single.parametrizations.weight.original"].astype(mx.bfloat16)
    assert mx.array_equal(out["single.weight"], expected)


def test_remap_reconstructs_weight_norm_pair():
    # Hand-built ground truth (independent of the implementation formula):
    # row0 v=(3,4) -> ||v||=5, g=10 -> w=(6, 8); row1 v=(0,2) -> ||v||=2, g=4 -> w=(0,4).
    weights = {
        "wn.parametrizations.weight.original0": mx.array([[10.0], [4.0]]),
        "wn.parametrizations.weight.original1": mx.array([[3.0, 4.0], [0.0, 2.0]]),
    }
    out = remap_state_dict(weights, mx.float32)
    expected = mx.array([[6.0, 8.0], [0.0, 4.0]])
    assert "wn.weight" in out
    assert mx.allclose(out["wn.weight"], expected, atol=1e-5)


def test_remap_splits_sonic_w13_into_gate_up_down():
    weights, (gate, up) = _make_state_dict()
    out = remap_state_dict(weights, mx.bfloat16)

    gate_key = f"layers.5.{SWITCH_MLP_PREFIX}.gate_proj.weight"
    up_key = f"layers.5.{SWITCH_MLP_PREFIX}.up_proj.weight"
    down_key = f"layers.5.{SWITCH_MLP_PREFIX}.down_proj.weight"

    assert gate_key in out and up_key in out and down_key in out
    # fused source keys are consumed
    assert "layers.5.feed_forward.experts.w13" not in out
    assert "layers.5.feed_forward.experts.w2" not in out

    assert out[gate_key].shape == (EXPERTS, INTER, HIDDEN)
    assert out[up_key].shape == (EXPERTS, INTER, HIDDEN)
    assert out[down_key].shape == (EXPERTS, HIDDEN, INTER)
    # even rows -> gate, odd rows -> up (element-wise vs hand-built tensors)
    assert mx.array_equal(out[gate_key], gate.astype(mx.bfloat16))
    assert mx.array_equal(out[up_key], up.astype(mx.bfloat16))


def test_remap_drops_training_only_router_stats():
    weights, _ = _make_state_dict()
    out = remap_state_dict(weights, mx.bfloat16)
    assert not any(k.endswith(".router.ent_denom") for k in out)
    assert not any(k.endswith(".router.normalized_entropy") for k in out)


def test_remap_passes_through_embeddings_head_dense_and_router():
    weights, _ = _make_state_dict()
    out = remap_state_dict(weights, mx.bfloat16)
    for key in (
        "multi_embedder.embedders.0.weight",
        "multi_embedder.embedders.8.weight",
        "multi_output.weight",
        "layers.0.feed_forward.w_in.weight",
        "layers.0.feed_forward.w_out.weight",
        "layers.5.feed_forward.router.down_proj.weight",
        "layers.5.feed_forward.router.router_mlp.4.weight",
        "layers.5.feed_forward.router.balancing_biases",
        "layers.5.feed_forward.router.router_states_scale",
    ):
        assert key in out, f"missing pass-through key {key}"
    # fused dense w_in keeps its rank-3 shape
    assert out["layers.0.feed_forward.w_in.weight"].shape == (2, INTER, HIDDEN)


def test_remap_dtype_policy():
    weights, _ = _make_state_dict()
    out = remap_state_dict(weights, mx.bfloat16)

    # weights -> bf16
    assert out["multi_output.weight"].dtype == mx.bfloat16
    assert out["layers.0.feed_forward.w_in.weight"].dtype == mx.bfloat16
    assert out[f"layers.5.{SWITCH_MLP_PREFIX}.gate_proj.weight"].dtype == mx.bfloat16
    assert out[f"layers.5.{SWITCH_MLP_PREFIX}.down_proj.weight"].dtype == mx.bfloat16

    # norms -> fp32
    assert out["layers.0.attention_norm.weight"].dtype == mx.float32
    assert out["layers.0.ffn_norm.weight"].dtype == mx.float32
    assert out["norm.weight"].dtype == mx.float32

    # router subtree + special scalars -> fp32
    assert out["layers.5.feed_forward.router.down_proj.weight"].dtype == mx.float32
    assert out["layers.5.feed_forward.router.router_mlp.4.weight"].dtype == mx.float32
    assert out["layers.5.feed_forward.router.balancing_biases"].dtype == mx.float32
    assert out["layers.5.feed_forward.router.router_states_scale"].dtype == mx.float32


# ── full convert() path (torch loader monkeypatched) ──────────────────────────


def test_convert_end_to_end_writes_safetensors(tmp_path, monkeypatch):
    weights, (gate, up) = _make_state_dict()

    # Bypass torch: feed the synthetic MLX dict straight into the pipeline.
    monkeypatch.setattr(convert_mod, "_load_torch_state_dict", lambda _pth: weights)
    monkeypatch.setattr(convert_mod, "_torch_state_dict_to_mx", lambda sd: dict(sd))

    # params.json -> config.json passthrough (+ dtype patch)
    src = tmp_path / "src"
    src.mkdir()
    (src / "model.pth").write_bytes(b"")  # presence only; loader is patched
    (src / "params.json").write_text('{"n_layers": 28, "dim": 2048}')

    out_dir = tmp_path / "out"
    returned = convert(src, out_dir, dtype="bfloat16")
    assert returned == out_dir

    st_path = out_dir / "model.safetensors"
    assert st_path.exists()

    reloaded = dict(mx.load(str(st_path)))
    gate_key = f"layers.5.{SWITCH_MLP_PREFIX}.gate_proj.weight"
    up_key = f"layers.5.{SWITCH_MLP_PREFIX}.up_proj.weight"

    # remap survived the safetensors round-trip
    assert gate_key in reloaded and up_key in reloaded
    assert "layers.5.feed_forward.experts.w13" not in reloaded
    assert not any(".parametrizations." in k for k in reloaded)
    assert mx.array_equal(reloaded[gate_key], gate.astype(mx.bfloat16))
    assert mx.array_equal(reloaded[up_key], up.astype(mx.bfloat16))

    # dtype policy survived
    assert reloaded["norm.weight"].dtype == mx.float32
    assert reloaded["multi_output.weight"].dtype == mx.bfloat16

    # config.json written from params.json with dtype patched
    import json

    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["n_layers"] == 28
    assert cfg["dtype"] == "bfloat16"


def test_convert_rejects_unknown_dtype(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        convert(tmp_path, tmp_path / "out", dtype="float8")


# ── _torch_state_dict_to_mx (fake torch tensors, no torch import needed) ───────


class _FakeTensor:
    """Minimal stand-in for a torch.Tensor exposing the methods the loader uses."""

    def __init__(self, np_array: np.ndarray, floating: bool):
        self._a = np_array
        self._floating = floating
        self.float_called = False

    def detach(self) -> "_FakeTensor":
        return self

    def to(self, _device: str) -> "_FakeTensor":
        return self

    def is_floating_point(self) -> bool:
        return self._floating

    def float(self) -> "_FakeTensor":
        self.float_called = True
        return _FakeTensor(self._a.astype(np.float32), True)

    def numpy(self) -> np.ndarray:
        return self._a


def test_torch_loader_upcasts_floats_and_preserves_integers():
    float_t = _FakeTensor(np.ones((2, 2), dtype=np.float16), floating=True)
    int_t = _FakeTensor(np.array([1025, 1024], dtype=np.int64), floating=False)

    out = _torch_state_dict_to_mx({"w": float_t, "pad_idx": int_t})

    # floating tensor went through .float() (bf16/fp16 cannot hit numpy directly)
    assert float_t.float_called is True
    assert out["w"].dtype == mx.float32
    # integer buffer kept its integer dtype and exact values (not floated)
    assert int_t.float_called is False
    assert out["pad_idx"].dtype == mx.int64
    assert mx.array_equal(out["pad_idx"], mx.array([1025, 1024], dtype=mx.int64))


# ── collision guard ───────────────────────────────────────────────────────────


def test_remap_raises_on_target_key_collision():
    import pytest

    # A pre-existing split key colliding with what w13 would produce.
    gate = mx.zeros((EXPERTS, INTER, HIDDEN))
    up = mx.zeros((EXPERTS, INTER, HIDDEN))
    weights = {
        "layers.5.feed_forward.experts.w13": _interleaved_w13(gate, up),
        f"layers.5.{SWITCH_MLP_PREFIX}.gate_proj.weight": mx.zeros(
            (EXPERTS, INTER, HIDDEN)
        ),
    }
    with pytest.raises(ValueError):
        remap_state_dict(weights, mx.bfloat16)
