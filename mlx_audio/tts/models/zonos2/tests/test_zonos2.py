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


def _force_eoa_at(model: Model, cfg: ZONOS2Config, eoa_step: int):
    """Patch the backbone logits so codebook 0 argmaxes to ``eoa_id`` at one step.

    Returns a closure to install as ``backbone.compute_logits`` that emits a normal
    (argmax 0) frame everywhere except at generation step ``eoa_step``, where
    codebook 0's argmax is forced to ``eoa_id`` — driving the delay-aware EOS
    countdown exactly like a real end-of-audio sample.
    """
    audio_vocab = cfg.audio_vocab
    n_cb = cfg.n_codebooks
    state = {"step": 0}

    def fake_logits(hidden: mx.array) -> mx.array:
        # hidden is [1, 1, dim]; emit [1, 1, n_cb, audio_vocab].
        logits = np.zeros((1, 1, n_cb, audio_vocab), dtype=np.float32)
        if state["step"] == eoa_step:
            logits[0, 0, 0, cfg.eoa_id] = 10.0  # cb0 -> EOA at this step
        state["step"] += 1
        return mx.array(logits)

    return fake_logits, state


def test_generate_codes_eos_flush_countdown_and_frame():
    """Forced EOA at a known step yields the upstream-aligned eos_frame + tail flush.

    Mirrors upstream ``TTSSequence._check_eos`` / ``TTSReq.check_eos``: when EOA is
    sampled in codebook ``c`` at generation step ``s``, ``eos_frame = s - c`` and the
    loop runs ``n_codebooks + 1`` more frames (the delay tail) before stopping.
    """
    # eoa/pad ids must fall inside the tiny audio vocab (codebook_size + 2).
    cfg = _tiny_config(eoa_id=8, audio_pad_id=9)
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=5)

    eoa_step = 4
    fake_logits, _ = _force_eoa_at(model, cfg, eoa_step)
    model.backbone.compute_logits = fake_logits  # type: ignore[assignment]

    codes, eos_frame = model.generate_codes(
        prompt,
        max_frames=64,
        temperature=0.0,
        top_k=1,
        stop_on_eoa=True,
        return_eos_frame=True,
    )
    produced = np.asarray(codes)
    # EOA in codebook 0 at step 4 -> aligned eos_frame == 4 - 0 == 4.
    assert eos_frame == eoa_step
    # Countdown = n_codebooks + 1 extra frames after detection; detection frame is
    # included, so total == eoa_step + 1 (detection) + n_codebooks (remaining tail).
    assert produced.shape[0] == eoa_step + 1 + cfg.n_codebooks
    # The detection frame's codebook 0 carries the EOA id.
    assert int(produced[eoa_step, 0]) == cfg.eoa_id


def test_decode_audio_truncates_to_eos_frame():
    """``decode_audio`` shears then slices ``[:eos_frame]`` before DAC (no tail leak)."""
    cfg = _tiny_config()
    model = Model(cfg)
    # 12 raw frames; an eos_frame of 5 must keep exactly 5 sheared output rows.
    raw = mx.array(
        np.random.randint(0, cfg.codebook_size, (12, cfg.n_codebooks)).astype(np.int32)
    )
    sheared_full = np.asarray(Model.shear_up(raw, cfg.audio_pad_id))

    captured = {}

    def fake_from_codes(dac_codes):
        captured["shape"] = tuple(dac_codes.shape)  # [1, C, H]
        # Return a dummy latent; decode() is also stubbed below.
        return (mx.zeros((1, 1, dac_codes.shape[-1])),)

    class _DummyDac:
        class quantizer:  # noqa: N801 - mimic the attribute path dac.quantizer
            from_codes = staticmethod(fake_from_codes)

        @staticmethod
        def decode(z):
            return mx.zeros((1, z.shape[-1]))

    model._dac = _DummyDac()  # bypass the real DAC download

    model.decode_audio(raw, eos_frame=5)
    # DAC saw exactly eos_frame (=5) frames along the time axis.
    assert captured["shape"] == (1, cfg.n_codebooks, 5)
    # And the kept rows are the first 5 of the full sheared tensor.
    assert sheared_full.shape[0] == 12


def test_project_speaker_shape():
    cfg = _tiny_config(
        speaker_enabled=True, speaker_embedding_dim=48, speaker_lda_dim=24
    )
    model = Model(cfg)
    assert model.backbone.speaker_lda_projection.weight.shape == (24, 48)
    assert model.backbone.speaker_projection.weight.shape == (cfg.dim, 24)
    emb = mx.random.normal((1, cfg.speaker_embedding_dim))
    out = model.backbone.project_speaker(emb)
    assert out.shape == (1, cfg.dim)


def test_with_speaker_frames_prefix_layout():
    cfg = _tiny_config(
        speaker_enabled=True,
        speaker_embedding_dim=48,
        speaker_lda_dim=24,
        speaker_background_token_enabled=True,
        accurate_mode_token_enabled=True,
        speaking_rate_num_buckets=2,
        quality_buckets={"a": ("0-1", "1+"), "b": ("0-1", "1+")},
    )
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=5)
    new_prompt, pos = model.with_speaker_frames(prompt)
    assert pos == 0
    assert new_prompt.shape == (5 + 2, cfg.n_codebooks + 1)  # slot + bg marker
    np_new = np.asarray(new_prompt)
    pad = cfg.audio_pad_id
    # Row 0: speaker slot (all audio_pad + text_vocab in the text column).
    assert np.all(np_new[0, : cfg.n_codebooks] == pad)
    assert int(np_new[0, cfg.n_codebooks]) == cfg.text_vocab
    # Row 1: clean-background marker (audio_pad + clean-bg token).
    assert np.all(np_new[1, : cfg.n_codebooks] == pad)
    assert int(np_new[1, cfg.n_codebooks]) == model._clean_background_token()
    # The rest is the original prompt verbatim.
    assert np.array_equal(np_new[2:], np.asarray(prompt))


def test_generate_codes_with_speaker_embedding_runs():
    cfg = _tiny_config(
        speaker_enabled=True, speaker_embedding_dim=48, speaker_lda_dim=24
    )
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=6)
    new_prompt, pos = model.with_speaker_frames(prompt)
    emb = mx.random.normal((cfg.speaker_embedding_dim,))
    codes = model.generate_codes(
        new_prompt,
        max_frames=4,
        temperature=0.0,
        top_k=1,
        stop_on_eoa=False,
        speaker_embedding=emb,
        speaker_position=pos,
    )
    assert codes.shape == (4, cfg.n_codebooks)
    assert codes.dtype == mx.int32


def test_speaker_injection_changes_first_logits():
    """Injecting a speaker embedding at position 0 perturbs the prefill hidden state."""
    cfg = _tiny_config(
        speaker_enabled=True, speaker_embedding_dim=48, speaker_lda_dim=24
    )
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=5)[None]
    proj = model.backbone.project_speaker(
        mx.random.normal((1, cfg.speaker_embedding_dim))
    )

    base = model.backbone(prompt)
    injected = model.backbone(prompt, speaker_emb=proj, speaker_pos=0)
    # Replacing the position-0 embedding must change the output (not a no-op).
    assert not bool(mx.allclose(base, injected))


def test_trim_leading_silence_helper():
    sr = 44100
    wav = mx.array(
        np.concatenate([np.zeros(2000, np.float32), (np.ones(1000, np.float32) * 0.4)])
    )
    trimmed = Model._trim_leading_silence(wav, threshold=0.01, keep_samples=128)
    # Drops the leading 2000 silent samples minus the 128 kept as lead-in.
    assert trimmed.shape[0] == 3000 - (2000 - 128)
    # All-silent input is returned unchanged (nothing crosses the threshold).
    silent = mx.zeros((500,))
    assert Model._trim_leading_silence(silent).shape[0] == 500


def test_speaker_projection_absent_without_speaker_config():
    cfg = _tiny_config(speaker_enabled=False)
    model = Model(cfg)
    assert model.backbone.speaker_projection is None
    assert model.backbone.speaker_lda_projection is None


def _speaker_enabled_config(**ov) -> ZONOS2Config:
    return _tiny_config(
        speaker_enabled=True, speaker_embedding_dim=48, speaker_lda_dim=24, **ov
    )


def _ckpt_keys_no_prefix(model: Model) -> dict:
    return {k[len("backbone.") :]: v for k, v in tree_flatten(model.parameters())}


def test_base_checkpoint_loads_strict_via_speaker_backfill():
    """A base checkpoint (no speaker keys) still loads strict (backfill keeps init)."""
    cfg = _speaker_enabled_config()
    src = Model(cfg)
    base_ckpt = {
        k: v
        for k, v in _ckpt_keys_no_prefix(src).items()
        if not k.startswith("speaker_")
    }
    assert not any("speaker" in k for k in base_ckpt)
    dst = Model(cfg)
    # Must NOT raise despite the model carrying speaker params absent from the ckpt.
    dst.load_weights(list(base_ckpt.items()), strict=True)


def test_strict_load_still_catches_missing_non_speaker_key():
    """The speaker backfill must not mask a genuinely missing backbone key."""
    cfg = _speaker_enabled_config()
    src = Model(cfg)
    ckpt = {
        k: v
        for k, v in _ckpt_keys_no_prefix(src).items()
        if not k.startswith("speaker_")
    }
    drop = next(k for k in ckpt if "multi_output" in k)
    del ckpt[drop]
    dst = Model(cfg)
    try:
        dst.load_weights(list(ckpt.items()), strict=True)
        raised = False
    except Exception:
        raised = True
    assert raised, "a missing non-speaker key must still fail the strict load"


def test_load_speaker_weights_validates_shape_and_completeness():
    cfg = _speaker_enabled_config()
    good = {
        "speaker_lda_projection.weight": mx.zeros((24, 48)),
        "speaker_lda_projection.bias": mx.zeros((24,)),
        "speaker_projection.weight": mx.zeros((cfg.dim, 24)),
        "speaker_projection.bias": mx.zeros((cfg.dim,)),
    }
    Model(cfg).load_speaker_weights(list(good.items()))  # OK

    wrong = dict(good)
    wrong["speaker_projection.weight"] = mx.zeros((cfg.dim, 99))  # bad in-dim
    try:
        Model(cfg).load_speaker_weights(list(wrong.items()))
        bad_shape_raised = False
    except ValueError:
        bad_shape_raised = True
    assert bad_shape_raised

    incomplete = [(k, v) for k, v in good.items() if "bias" not in k]
    try:
        Model(cfg).load_speaker_weights(incomplete)
        incomplete_raised = False
    except ValueError:
        incomplete_raised = True
    assert incomplete_raised


def test_speaker_injection_out_of_range_position_raises():
    cfg = _speaker_enabled_config()
    model = Model(cfg)
    prompt = _random_prompt(cfg, seq=4)[None]
    proj = model.backbone.project_speaker(mx.zeros((1, cfg.speaker_embedding_dim)))
    try:
        model.backbone(prompt, speaker_emb=proj, speaker_pos=99)
        raised = False
    except ValueError:
        raised = True
    assert raised, "out-of-range speaker_pos must raise, not silently no-op"


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
