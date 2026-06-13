"""Convert Zyphra ZONOS2 (sparse-MoE multi-codebook AR TTS) weights to MLX.

Loads the upstream PyTorch ``model.pth`` state-dict, remaps keys to the MLX
layout used by this port, applies the bf16/fp32 dtype policy, and writes
``model.safetensors`` (+ ``config.json`` derived from ``params.json`` if present).

The key remapping mirrors the upstream loaders:
  - ``python/zonos2/models/weight.py``  -> ``_normalize_zonos2_state_dict``
    (weight-norm parametrization stripping, drops training-only router stats).
  - ``python/zonos2/models/zonos2.py``  -> ``FusedGroupedExperts.load_state_dict``
    / ``_convert_sonic_w13_to_gate_up`` (de-interleave the fused SonicMoE ``w13``).

Unlike upstream, which keeps the experts FUSED as ``gate_up_proj``, this port
targets ``mlx_lm.models.switch_layers.SwitchGLU`` which expects SEPARATE
``gate_proj`` / ``up_proj`` / ``down_proj`` stacked-expert weights, so the
SonicMoE ``w13`` is split into two tensors rather than re-concatenated.

The remap core (:func:`remap_state_dict`, :func:`split_sonic_w13`) operates on a
plain ``dict[str, mx.array]`` so it can be unit-tested without ``torch``; the
``torch`` dependency is imported lazily inside :func:`convert` only.

Usage:
    python -m mlx_audio.tts.models.zonos2.convert \\
        --model /path/to/zonos2 \\
        --output ./models/zonos2-mlx \\
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx

# ── target MLX key layout (single source of truth for the integration layer) ──
#
# Dense FFN (layers 0/1/2/27):   layers.{N}.feed_forward.w_in.weight  [2, inter, hidden]
#                                layers.{N}.feed_forward.w_out.weight [hidden, inter]
# MoE experts (SonicMoE, 3..26): de-interleaved into SwitchGLU stacked weights ->
#   layers.{N}.feed_forward.switch_mlp.gate_proj.weight  [E, inter, hidden]
#   layers.{N}.feed_forward.switch_mlp.up_proj.weight    [E, inter, hidden]
#   layers.{N}.feed_forward.switch_mlp.down_proj.weight  [E, hidden, inter]
SWITCH_MLP_PREFIX = "feed_forward.switch_mlp"

# Fused SonicMoE experts checkpoint key suffixes (relative to the FFN prefix).
SONIC_W13_SUFFIX = "feed_forward.experts.w13"
SONIC_W2_SUFFIX = "feed_forward.experts.w2"

# Training-only router statistics that upstream drops for inference.
_ROUTER_TRAINING_ONLY = (".router.ent_denom", ".router.normalized_entropy")

# Tensors kept in fp32 for numerical parity (norms, router, scales/biases).
_FP32_SUBSTRINGS = (
    "norm",  # *_norm.weight, rmsnorm*, attention_norm, ffn_norm, final norm
    "router.",  # the entire router sub-tree
    "balancing_biases",
    "router_states_scale",
)


# ── MoE de-interleave (mirror upstream _convert_sonic_w13_to_gate_up) ─────────


def split_sonic_w13(w13: mx.array) -> Tuple[mx.array, mx.array]:
    """De-interleave a fused SonicMoE ``w13`` into separate gate/up projections.

    Upstream fuses the SwiGLU input projection as a single rank-3 tensor
    ``[experts, 2 * intermediate, hidden]`` with gate and up rows INTERLEAVED:
    even rows are the gate (SiLU branch), odd rows are the up branch. Mirrors
    ``zonos2.models.zonos2._convert_sonic_w13_to_gate_up`` (which then re-concats
    them; here we keep them separate for ``SwitchGLU``).

    Returns:
        ``(gate, up)`` each shaped ``[experts, intermediate, hidden]``.
    """
    if w13.ndim != 3:
        raise ValueError(f"Expected SonicMoE w13 to be rank-3, got shape {w13.shape}")
    if w13.shape[1] % 2 != 0:
        raise ValueError(f"Expected even fused width in SonicMoE w13, got {w13.shape}")
    gate = w13[:, 0::2, :]
    up = w13[:, 1::2, :]
    return gate, up


# ── weight-norm parametrization stripping ─────────────────────────────────────


def _strip_weight_norm(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Collapse ``*.parametrizations.X.original*`` keys back to plain ``*.X``.

    Mirrors ``zonos2.models.weight._normalize_zonos2_state_dict`` for the common
    case where the parametrization stores a single ``.original`` (a straight
    rename: ``a.parametrizations.weight.original`` -> ``a.weight``).

    Additionally reconstructs the modern two-tensor
    ``torch.nn.utils.parametrizations.weight_norm`` form when BOTH
    ``original0`` (magnitude ``g``) and ``original1`` (direction ``v``) are
    present: ``w = g * v / ||v||`` with the norm taken over every dim except 0.
    """
    result: Dict[str, mx.array] = {}
    pending_pairs: Dict[str, Dict[str, mx.array]] = {}

    for key, value in weights.items():
        if ".parametrizations." not in key or ".original" not in key:
            result[key] = value
            continue

        base = key.replace(".parametrizations.", ".")
        if base.endswith(".original0") or base.endswith(".original1"):
            target = base[: -len(".original0")]  # drop ".originalN"
            slot = "g" if base.endswith(".original0") else "v"
            pending_pairs.setdefault(target, {})[slot] = value
        elif base.endswith(".original"):
            # Single ".original" -> strip only the trailing marker (anchored, so
            # a path segment that merely contains "original" is left intact).
            result[base[: -len(".original")]] = value
        else:
            result[base] = value

    for target, parts in pending_pairs.items():
        if "g" in parts and "v" in parts:
            result[target] = _reconstruct_weight_norm(parts["g"], parts["v"])
        else:
            # Only one half present: keep it under the collapsed name so nothing
            # is silently dropped (lets the caller notice the malformed pair).
            (only,) = parts.values()
            result[target] = only

    return result


def _reconstruct_weight_norm(g: mx.array, v: mx.array) -> mx.array:
    """Recompute the effective weight from weight-norm parts (``dim=0`` default).

    ``g`` is the per-output-channel magnitude ``[out, 1, ...]``; ``v`` is the
    direction with the full weight shape. The L2 norm of ``v`` is taken over all
    dims except 0, matching PyTorch's ``_WeightNorm`` forward.
    """
    reduce_axes = tuple(range(1, v.ndim))
    v32 = v.astype(mx.float32)
    norm = mx.sqrt(mx.sum(v32 * v32, axis=reduce_axes, keepdims=True))
    # Return fp32; the caller's dtype policy decides the final precision.
    return g.astype(mx.float32) * (v32 / norm)


# ── dtype policy ──────────────────────────────────────────────────────────────


def _target_dtype(key: str, weight_dtype: mx.Dtype) -> mx.Dtype:
    """Norms / router / scale-and-bias tensors stay fp32; everything else casts."""
    if any(sub in key for sub in _FP32_SUBSTRINGS):
        return mx.float32
    return weight_dtype


# ── core remap (pure MLX, torch-free, unit-testable) ─────────────────────────


def remap_state_dict(
    weights: Dict[str, mx.array], weight_dtype: mx.Dtype
) -> Dict[str, mx.array]:
    """Remap an MLX-array ZONOS2 state-dict to this port's layout + dtype policy.

    Steps (mirroring the upstream loaders):
      1. Strip weight-norm parametrization (``*.parametrizations.X.original*``).
      2. Drop training-only router statistics.
      3. De-interleave each MoE ``feed_forward.experts.w13`` into separate
         ``switch_mlp.gate_proj``/``up_proj`` weights and rename ``w2`` ->
         ``switch_mlp.down_proj`` (dense ``feed_forward.w_in``/``w_out`` are
         passed through unchanged).
      4. Apply the bf16/fp32 dtype policy.
    """
    weights = _strip_weight_norm(weights)

    remapped: Dict[str, mx.array] = {}

    def _put(target: str, value: mx.array) -> None:
        if target in remapped:
            raise ValueError(f"Duplicate target key after remap: {target!r}")
        remapped[target] = value

    for key, value in weights.items():
        if any(stat in key for stat in _ROUTER_TRAINING_ONLY):
            continue

        if key.endswith(SONIC_W13_SUFFIX):
            prefix = key[: -len(SONIC_W13_SUFFIX)]  # e.g. "layers.5."
            gate, up = split_sonic_w13(value)
            _put(f"{prefix}{SWITCH_MLP_PREFIX}.gate_proj.weight", gate)
            _put(f"{prefix}{SWITCH_MLP_PREFIX}.up_proj.weight", up)
            continue

        if key.endswith(SONIC_W2_SUFFIX):
            prefix = key[: -len(SONIC_W2_SUFFIX)]
            _put(f"{prefix}{SWITCH_MLP_PREFIX}.down_proj.weight", value)
            continue

        _put(key, value)

    return {k: v.astype(_target_dtype(k, weight_dtype)) for k, v in remapped.items()}


# ── torch -> MLX loading ──────────────────────────────────────────────────────


def _torch_state_dict_to_mx(state_dict: Dict[str, object]) -> Dict[str, mx.array]:
    """Convert a torch state-dict to ``{key: mx.array}``.

    Floating tensors are upcast to fp32 before ``.numpy()`` because numpy has no
    native bf16 (the real per-key dtype is applied later by the dtype policy).
    Integer / bool tensors keep their native dtype so index/pad buffers are not
    corrupted into floats.
    """
    out: Dict[str, mx.array] = {}
    for key, tensor in state_dict.items():
        tensor = tensor.detach().to("cpu")
        if tensor.is_floating_point():
            tensor = tensor.float()
        out[key] = mx.array(tensor.numpy())
    return out


def _load_torch_state_dict(pth_path: Path) -> Dict[str, object]:
    import torch  # lazy: torch is only needed for real conversion, not tests

    obj = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


# ── config.json from params.json ──────────────────────────────────────────────


def _write_config(src_dir: Path, out_dir: Path, dtype: str) -> None:
    """Copy ``params.json`` (raw ZONOS2 field names) to ``config.json``, set dtype.

    ``ZONOS2Config.from_dict`` filters to its known fields, so the raw
    ``params.json`` field names are the source of truth and pass through as-is.
    """
    params_path = src_dir / "params.json"
    if not params_path.exists():
        print(
            f"  WARNING: no params.json next to the checkpoint ({src_dir}); "
            "config.json was NOT written — the MLX model dir will be incomplete"
        )
        return
    with open(params_path) as f:
        params = json.load(f)
    params["dtype"] = dtype
    with open(out_dir / "config.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"  wrote config.json from {params_path.name}")


_DTYPE_MAP = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def convert(pth_path: str | Path, out_dir: str | Path, dtype: str = "bfloat16") -> Path:
    """Convert a ZONOS2 ``model.pth`` to an MLX ``model.safetensors`` directory.

    Args:
        pth_path: path to the torch ``model.pth`` (or its parent directory).
        out_dir: output directory for ``model.safetensors`` (+ ``config.json``).
        dtype: target weight dtype (``"bfloat16"`` / ``"float16"`` / ``"float32"``);
            norms/router/scale tensors are always written as fp32.

    Returns:
        The output directory as a ``Path``.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; choose one of {list(_DTYPE_MAP)}"
        )
    weight_dtype = _DTYPE_MAP[dtype]

    pth_path = Path(pth_path)
    src_dir = pth_path.parent if pth_path.is_file() else pth_path
    if pth_path.is_dir():
        pth_path = pth_path / "model.pth"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading torch checkpoint {pth_path}…")
    torch_state = _load_torch_state_dict(pth_path)
    weights = _torch_state_dict_to_mx(torch_state)
    print(f"  {len(weights)} tensors")

    weights = remap_state_dict(weights, weight_dtype)

    out_path = out_dir / "model.safetensors"
    mx.save_safetensors(str(out_path), weights)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  saved {out_path.name} ({size_mb:.0f} MB, {len(weights)} tensors)")

    _write_config(src_dir, out_dir, dtype)

    print(f"Done → {out_dir}")
    return out_dir


# ── Qwen3 voice embedder conversion ───────────────────────────────────────────


def convert_voice_embedder(
    hf_repo: str = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
    out_dir: str | Path = "./models/zonos2-voice-embedder-mlx",
    dtype: str = "bfloat16",
) -> Path:
    """Convert the Qwen3-1.7B voice-embedder backbone (HF -> MLX safetensors).

    The mlx-audio ``qwen3`` model reuses ``mlx_lm.models.qwen3`` (``Model`` /
    ``ModelArgs``), so the HF -> MLX conversion is exactly the standard mlx_lm
    qwen3 path (sanitize = drop tied ``lm_head.weight``). This delegates to
    ``mlx_lm.convert`` rather than re-implementing the qwen3 sanitize logic.

    NOTE: the voice embedder uses the qwen3 backbone hidden states (2048-D) as a
    speaker embedding; the embedding-head / mel-frontend wiring is finalized by
    the coordinator in ``speaker_encoder.py`` (CONTRACT.md §5). This produces the
    plain MLX-format qwen3 weights that the embedder loads.

    Args:
        hf_repo: HuggingFace repo id for the voice-embedding Qwen3 backbone.
        out_dir: output directory for the MLX model.
        dtype: target weight dtype.

    Returns:
        The output directory as a ``Path``.
    """
    from mlx_lm import convert as mlx_lm_convert

    out_dir = Path(out_dir)
    print(f"Converting voice embedder {hf_repo} → {out_dir} (dtype={dtype})…")
    mlx_lm_convert(hf_repo, str(out_dir), dtype=dtype)
    print(f"Done → {out_dir}")
    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ZONOS2 to mlx-audio format")
    parser.add_argument(
        "--model", required=True, help="Path to model.pth or its parent directory"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=list(_DTYPE_MAP),
        help="Target weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--voice-embedder",
        default=None,
        metavar="HF_REPO",
        help="Also convert the Qwen3 voice embedder from this HF repo into "
        "<output>/voice_embedder",
    )
    args = parser.parse_args()

    out = convert(args.model, args.output, args.dtype)

    if args.voice_embedder:
        convert_voice_embedder(args.voice_embedder, out / "voice_embedder", args.dtype)


if __name__ == "__main__":
    main()
