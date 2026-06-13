# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""Wiring tests for the ZONOS2 top-level :class:`Model` on a tiny synthetic config.

These exercise the integration glue (multi-embedder sum, fused add-norm residual
threading, MoE/dense layer mix, output head reshape + softcap, the greedy AR loop
with the delay-aware next-frame construction, ``shear_up``, and checkpoint-key
reconciliation) without needing the real 8B checkpoint.
"""

from __future__ import annotations

import re

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.tts.models.zonos2.config import ZONOS2Config
from mlx_audio.tts.models.zonos2.zonos2 import (
    Model,
    MultiEmbedding,
    RMSNormFused,
    Zonos2Backbone,
)


def _tiny_config(**overrides) -> ZONOS2Config:
    base = dict(
        n_layers=6,
        dim=64,
        head_dim=16,
        n_kv_heads=2,
        n_codebooks=9,
        codebook_size=8,
        text_vocab=12,
        moe_n_experts=4,
        moe_router_dim=16,
        moe_start_from_layer=3,
        moe_end_from_layer=1,
        special_topk_layers={4: 2},
        ffn_dim_multiplier=1.5,
        multiple_of=16,
        rope_theta=10000.0,
    )
    base.update(overrides)
    return ZONOS2Config(**base)


def _random_prompt(cfg: ZONOS2Config, seq: int) -> mx.array:
    w = cfg.n_codebooks + 1
    ids = np.zeros((seq, w), dtype=np.int32)
    # audio columns in [0, codebook_size+1], text column in [0, text_vocab]
    ids[:, : cfg.n_codebooks] = np.random.randint(
        0, cfg.audio_vocab, (seq, cfg.n_codebooks)
    )
    ids[:, cfg.n_codebooks] = np.random.randint(0, cfg.text_vocab + 1, seq)
    return mx.array(ids)


def test_rmsnorm_fused_residual_threading():
    norm = RMSNormFused(8, eps=1e-5)  # default: output dtype tracks input
    x = mx.random.normal((1, 3, 8)).astype(mx.bfloat16)
    # First call: residual is None -> running residual is the raw input (fp32).
    out0, res0 = norm(x, None)
    assert res0.dtype == mx.float32  # residual stream stays fp32 for parity
    assert mx.allclose(res0, x.astype(mx.float32))
    assert out0.shape == x.shape
    assert out0.dtype == mx.bfloat16  # normed output tracks the bf16 input
    # Second call: residual accumulates (res = res + x) in fp32 before norm.
    y = mx.random.normal((1, 3, 8)).astype(mx.bfloat16)
    out1, res1 = norm(y, res0)
    assert res1.dtype == mx.float32
    assert mx.allclose(res1, res0 + y.astype(mx.float32))
    expected = mx.fast.rms_norm(
        res0 + y.astype(mx.float32), norm.weight.astype(mx.float32), 1e-5
    ).astype(mx.bfloat16)
    assert mx.allclose(out1, expected, atol=1e-3)


def test_rmsnorm_fused_residual_stays_fp32_for_fp32_input():
    norm = RMSNormFused(8, eps=1e-5)
    x = mx.random.normal((1, 3, 8))  # fp32 input -> fp32 output (dtype tracked)
    out, res = norm(x, None)
    assert out.dtype == mx.float32 and res.dtype == mx.float32


def test_rmsnorm_fused_compute_dtype_override():
    norm = RMSNormFused(8, eps=1e-5, compute_dtype=mx.float32)
    x = mx.random.normal((1, 3, 8)).astype(mx.bfloat16)
    out, res = norm(x, None)
    # out_norm-style override forces fp32 output regardless of input dtype.
    assert out.dtype == mx.float32 and res.dtype == mx.float32


def test_rmsnorm_fused_no_affine_has_no_param():
    norm = RMSNormFused(8, elementwise_affine=False)
    keys = [k for k, _ in tree_flatten(norm.parameters())]
    assert keys == []  # unit weight is a constant, not a checkpoint parameter


def test_multi_embedding_is_sum_over_columns():
    cfg = _tiny_config()
    emb = MultiEmbedding(cfg)
    ids = _random_prompt(cfg, seq=4)
    out = emb(ids)
    assert out.shape == (4, cfg.dim)
    # Reconstruct the sum manually from each per-column table.
    manual = emb.embedders[0](ids[..., 0])
    for i in range(1, ids.shape[-1]):
        manual = manual + emb.embedders[i](ids[..., i])
    assert mx.allclose(out, manual)
    assert len(emb.embedders) == cfg.n_codebooks + 1


def test_backbone_logits_shape_and_softcap():
    cfg = _tiny_config()
    bb = Zonos2Backbone(cfg)
    ids = _random_prompt(cfg, seq=5)[None]
    hidden = bb(ids)
    assert hidden.shape == (1, 5, cfg.dim)
    logits = bb.compute_logits(hidden)
    assert logits.shape == (1, 5, cfg.n_codebooks, cfg.audio_vocab)
    # softcap bounds logits within (-cap, cap).
    assert bool(mx.all(mx.abs(logits) <= cfg.loss_softcap).item())


def test_model_generate_codes_shape_and_dtype():
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=7)
    codes = model.generate_codes(prompt, max_frames=5, temperature=0.0)
    assert codes.shape == (5, cfg.n_codebooks)
    assert codes.dtype == mx.int32


def test_model_generate_codes_is_deterministic_greedy():
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=6)
    a = np.asarray(model.generate_codes(prompt, max_frames=8, temperature=0.0))
    b = np.asarray(model.generate_codes(prompt, max_frames=8, temperature=0.0))
    assert np.array_equal(a, b)


def test_incremental_decode_matches_full_prefill():
    """One greedy step on a [seq] prompt == argmax of a full forward's last pos."""
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=6)

    # Reference: full prefill, argmax at last position.
    cache = model.backbone.make_cache()
    hidden = model.backbone(prompt[None], cache=cache)
    logits = model.backbone.compute_logits(hidden[:, -1:, :])
    expected = np.asarray(mx.argmax(logits[:, 0], axis=-1)[0]).astype(np.int64)

    produced = np.asarray(model.generate_codes(prompt, max_frames=1, temperature=0.0))[
        0
    ]
    assert np.array_equal(produced, expected)


def test_shear_up_removes_delay_pattern():
    # column j shifted up by j rows; vacated tail filled with pad_id.
    pad = 99
    codes = mx.array(
        [
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ],
        dtype=mx.int32,
    )
    out = np.asarray(Model.shear_up(codes, pad))
    expected = np.array(
        [
            [10, 21, 32],
            [20, 31, pad],
            [30, pad, pad],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(out, expected)


def test_next_input_row_uses_text_pad():
    cfg = _tiny_config()
    model = Model(cfg)
    codes = mx.arange(cfg.n_codebooks).astype(mx.int32)
    row = model._next_input_row(codes)
    assert row.shape == (1, 1, cfg.n_codebooks + 1)
    row_np = np.asarray(row[0, 0])
    assert np.array_equal(row_np[: cfg.n_codebooks], np.arange(cfg.n_codebooks))
    assert int(row_np[cfg.n_codebooks]) == cfg.text_vocab  # text pad id


def test_generate_codes_returns_eos_frame_when_requested():
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=5)
    out = model.generate_codes(
        prompt, max_frames=3, temperature=0.0, return_eos_frame=True
    )
    assert isinstance(out, tuple) and len(out) == 2
    codes, eos_frame = out
    assert codes.shape == (3, cfg.n_codebooks)
    # No EOA is forced here; eos_frame may be None or a valid int index.
    assert eos_frame is None or isinstance(eos_frame, int)


def test_teacher_forced_logits_shape():
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=5)
    forced = mx.array(
        np.random.randint(0, cfg.audio_vocab, (4, cfg.n_codebooks)).astype(np.int32)
    )
    logits = model.teacher_forced_logits(prompt, forced)
    assert logits.shape == (4, cfg.n_codebooks, cfg.audio_vocab)
    assert logits.dtype == mx.float32


def test_teacher_forced_logits_match_forced_replay():
    """Teacher-forced step-t logits == a manual forced KV-cache replay.

    Feeding ``forced_tokens`` step by step through the backbone (prompt last row,
    then each forced row) must reproduce exactly the per-step logits the method
    records, confirming the decode context is pinned to the forced tokens.
    """
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=6)
    n = 4
    forced_np = np.random.randint(0, cfg.audio_vocab, (n, cfg.n_codebooks)).astype(
        np.int32
    )
    forced = mx.array(forced_np)

    tf_logits = np.asarray(model.teacher_forced_logits(prompt, forced))

    # Manual replay: same prefill, then read logits BEFORE appending each forced
    # row (so step t is conditioned on prompt + forced[:t]).
    cache = model.backbone.make_cache()
    pids = prompt[None].astype(mx.int32)
    model.backbone(pids[:, :-1, :], cache=cache)
    hidden = model.backbone(pids[:, -1:, :], cache=cache)
    manual = []
    for t in range(n):
        logits = model.backbone.compute_logits(hidden[:, -1:, :])
        manual.append(np.asarray(logits[0, 0]))
        row = model._next_input_row(forced[t])
        hidden = model.backbone(row, cache=cache)
    manual = np.stack(manual, axis=0)
    assert np.allclose(tf_logits, manual, atol=1e-4)


def test_teacher_forced_logits_empty_returns_empty():
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=4)
    empty = mx.zeros((0, cfg.n_codebooks), dtype=mx.int32)
    logits = model.teacher_forced_logits(prompt, empty)
    assert logits.shape == (0, cfg.n_codebooks, cfg.audio_vocab)


def test_teacher_forced_with_own_greedy_tokens_is_consistent():
    """Forcing the model's own greedy tokens -> step-t argmax == that token."""
    cfg = _tiny_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=5)
    greedy = model.generate_codes(prompt, max_frames=4, temperature=0.0)
    logits = model.teacher_forced_logits(prompt, greedy)
    tf_argmax = np.asarray(mx.argmax(logits, axis=-1)).astype(np.int64)
    assert np.array_equal(tf_argmax, np.asarray(greedy).astype(np.int64))


def test_load_weights_reconciles_convert_keys():
    """A synthetic checkpoint in convert.py's key layout loads strictly."""
    cfg = _tiny_config()
    model = Model(cfg)

    def to_ckpt(key: str) -> str:
        key = key[len("backbone.") :]
        key = key.replace("feed_forward.experts.", "feed_forward.switch_mlp.")
        key = re.sub(r"(\.router\.router_mlp)\.layers\.(\d+)\.", r"\1.\2.", key)
        return key

    ckpt = {}
    for k, v in tree_flatten(model.parameters()):
        ckpt[to_ckpt(k)] = mx.random.normal(v.shape) if v.size > 0 else v

    # No raw `experts.`/`router_mlp.layers.` keys remain in the checkpoint view.
    assert any("switch_mlp" in k for k in ckpt)
    assert all(".router_mlp.layers." not in k for k in ckpt)

    fresh = Model(cfg)
    fresh.load_weights(list(ckpt.items()), strict=True)
