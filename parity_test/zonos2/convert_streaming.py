#!/usr/bin/env python
"""Memory-bounded ZONOS2 ``model.pth`` -> MLX safetensors conversion.

The released checkpoint is ~14.6 GB; loading it fully and materializing fp32
copies of every tensor at once can exceed this machine's shared-memory budget.
This converter mmaps the torch checkpoint (``torch.load(mmap=True)``) and
processes ONE tensor at a time -- converting, remapping (de-interleaving the
fused SonicMoE ``w13``, dropping speaker-conditioning + training-only keys),
casting per the bf16/fp32 policy, and freeing each torch tensor immediately --
so peak resident RAM stays a few GB plus the accumulating bf16 output dict.

It reuses the pure remap helpers from ``convert.py`` (single source of truth for
key names and the dtype policy) and only changes the *loading/streaming*
strategy. Speaker keys are dropped because the offline parity reference is
speaker-unconditioned (the backbone has no speaker projection).

Usage:
    uv run --no-sync --with torch --project /Volumes/DATA/mlx-audio python \
        parity_test/zonos2/convert_streaming.py \
        --model /Volumes/DATA/zonos2-ckpt/model.pth \
        --output /Volumes/DATA/zonos2-mlx
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Ensure the local worktree package shadows any installed mlx_audio so the
# in-progress zonos2 module is importable regardless of the launch cwd.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

import mlx.core as mx

from mlx_audio.tts.models.zonos2.convert import (
    _ROUTER_TRAINING_ONLY,
    SONIC_W2_SUFFIX,
    SONIC_W13_SUFFIX,
    SWITCH_MLP_PREFIX,
    _target_dtype,
    split_sonic_w13,
)

# mlx materialization (aliased to keep peak transient buffers bounded).
_materialize = mx.eval

# Speaker-conditioning keys: present in the checkpoint but unused by the
# unconditioned offline forward (no speaker projection in the backbone).
_SPEAKER_PREFIXES = ("speaker_lda_projection.", "speaker_projection.")

_DTYPE_MAP = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _rss_gb() -> float:
    try:
        import resource

        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024**3 if rss > 10**7 else 1024**2)
    except Exception:
        return -1.0


def _to_mx(tensor) -> mx.array:
    tensor = tensor.detach().to("cpu")
    if tensor.is_floating_point():
        tensor = tensor.float()
    return mx.array(tensor.numpy())


def convert_streaming(pth_path: str, out_dir: str, dtype: str = "bfloat16") -> Path:
    import torch

    weight_dtype = _DTYPE_MAP[dtype]
    pth_path = Path(pth_path)
    src_dir = pth_path.parent if pth_path.is_file() else pth_path
    if pth_path.is_dir():
        pth_path = pth_path / "model.pth"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mmap-loading torch checkpoint {pth_path} ...", flush=True)
    obj = torch.load(str(pth_path), map_location="cpu", weights_only=False, mmap=True)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state = obj["model"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    keys = list(state.keys())
    print(f"  {len(keys)} tensors; peak RSS so far {_rss_gb():.2f} GB", flush=True)

    out: dict[str, mx.array] = {}
    n_dropped = 0
    for i, key in enumerate(keys):
        if any(key.startswith(p) for p in _SPEAKER_PREFIXES):
            n_dropped += 1
            continue
        if any(stat in key for stat in _ROUTER_TRAINING_ONLY):
            n_dropped += 1
            continue

        value = _to_mx(state[key])

        if key.endswith(SONIC_W13_SUFFIX):
            prefix = key[: -len(SONIC_W13_SUFFIX)]
            gate, up = split_sonic_w13(value)
            gk = f"{prefix}{SWITCH_MLP_PREFIX}.gate_proj.weight"
            uk = f"{prefix}{SWITCH_MLP_PREFIX}.up_proj.weight"
            out[gk] = gate.astype(_target_dtype(gk, weight_dtype))
            out[uk] = up.astype(_target_dtype(uk, weight_dtype))
            _materialize(out[gk], out[uk])
            del gate, up
        elif key.endswith(SONIC_W2_SUFFIX):
            prefix = key[: -len(SONIC_W2_SUFFIX)]
            tgt = f"{prefix}{SWITCH_MLP_PREFIX}.down_proj.weight"
            out[tgt] = value.astype(_target_dtype(tgt, weight_dtype))
            _materialize(out[tgt])
        else:
            out[key] = value.astype(_target_dtype(key, weight_dtype))
            _materialize(out[key])

        del value
        if (i + 1) % 64 == 0:
            gc.collect()
            print(
                f"  [{i + 1}/{len(keys)}] out={len(out)} RSS={_rss_gb():.2f} GB",
                flush=True,
            )

    del state, obj
    gc.collect()
    print(f"  dropped {n_dropped} keys; writing {len(out)} tensors ...", flush=True)
    print(f"  peak RSS before save {_rss_gb():.2f} GB", flush=True)

    out_path = out_dir / "model.safetensors"
    mx.save_safetensors(str(out_path), out)
    size_gb = out_path.stat().st_size / 1e9
    print(f"  saved {out_path.name} ({size_gb:.2f} GB, {len(out)} tensors)", flush=True)

    params_path = src_dir / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        params["dtype"] = dtype
        (out_dir / "config.json").write_text(json.dumps(params, indent=2))
        print("  wrote config.json", flush=True)

    print(f"  final peak RSS {_rss_gb():.2f} GB", flush=True)
    print(f"Done -> {out_dir}", flush=True)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/Volumes/DATA/zonos2-ckpt/model.pth")
    ap.add_argument("--output", default="/Volumes/DATA/zonos2-mlx")
    ap.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP))
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    convert_streaming(args.model, args.output, args.dtype)


if __name__ == "__main__":
    main()
