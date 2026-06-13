#!/usr/bin/env python
"""ZONOS2 MLX end-to-end parity check against the captured CUDA reference.

For each reference item this:
  * loads the converted MLX model,
  * feeds the EXACT ``prompt_ids`` (bypassing the tokenizer to isolate backbone
    parity),
  * runs deterministic greedy decoding (argmax / topk=1, no repetition penalty)
    for ``n_frames`` steps with the same delay handling as upstream,
  * compares the produced frames to the reference ``audio_tokens`` and reports
    exact-match %, the first divergence (frame, codebook), and per-codebook
    agreement.

Greedy is deterministic across frameworks when the logits match, so a passing
run is frame-exact. On divergence the first mismatching (frame, codebook) and a
short per-codebook breakdown are printed so the backbone can be debugged.

Usage:
    uv run --no-sync --project /Volumes/DATA/mlx-audio python \
        parity_test/zonos2/check_full.py \
        --model /Volumes/DATA/zonos2-mlx \
        --reference /Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2/reference \
        [--decode-audio]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the local worktree package shadows any installed mlx_audio so the
# in-progress zonos2 module is importable regardless of the launch cwd.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

import mlx.core as mx

# Hold MLX/Metal under the machine's shared-memory budget (32 GB total; cap
# 20 GB). The bf16 8B weights (~15 GB) mmap lazily; activations stay small.
try:
    mx.set_memory_limit(int(18e9))
except Exception:
    pass
try:
    mx.set_cache_limit(int(1e9))
except Exception:
    pass
try:
    mx.set_wired_limit(int(18e9))
except Exception:
    pass

import numpy as np

from mlx_audio.tts.models.zonos2.zonos2 import Model

DEFAULT_REFERENCE = "/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2/reference"


def _flush(x: mx.array) -> mx.array:
    """Force materialization of an MLX array (avoid the name ``eval``)."""
    mx.async_eval(x)
    return x


def _per_codebook_agreement(produced: np.ndarray, reference: np.ndarray) -> list:
    n = min(produced.shape[0], reference.shape[0])
    p, r = produced[:n], reference[:n]
    return [float((p[:, c] == r[:, c]).mean()) for c in range(r.shape[1])]


def _first_divergence(produced: np.ndarray, reference: np.ndarray):
    n = min(produced.shape[0], reference.shape[0])
    for f in range(n):
        for c in range(reference.shape[1]):
            if int(produced[f, c]) != int(reference[f, c]):
                return f, c, int(produced[f, c]), int(reference[f, c])
    if produced.shape[0] != reference.shape[0]:
        return (n, -1, produced.shape[0], reference.shape[0])
    return None


def check_item(model: Model, ref_dir: Path, item: dict, decode_audio: bool):
    data = np.load(ref_dir / item["npz"])
    prompt_ids = mx.array(data["prompt_ids"].astype(np.int32))
    reference = data["audio_tokens"].astype(np.int64)
    n_frames = int(item["n_frames"])

    produced = model.generate_codes(
        prompt_ids,
        max_frames=n_frames,
        temperature=0.0,  # greedy
        top_k=1,
        stop_on_eoa=False,  # match the fixed-length reference capture
    )
    produced = np.asarray(produced).astype(np.int64)

    n = min(produced.shape[0], reference.shape[0])
    exact = int((produced[:n] == reference[:n]).all(axis=1).sum())
    total = reference.shape[0]
    div = _first_divergence(produced, reference)
    per_cb = _per_codebook_agreement(produced, reference)

    print(f"\n=== item {item['index']}: {item['text']!r} ===")
    print(f"  prompt_ids: {tuple(data['prompt_ids'].shape)}  n_frames: {n_frames}")
    print(f"  produced: {produced.shape}  reference: {reference.shape}")
    print(f"  EXACT FRAME MATCH: {exact}/{total}")
    if div is None:
        print("  ALL FRAMES EXACT")
    else:
        f, c, pv, rv = div
        if c == -1:
            print(f"  LENGTH MISMATCH: produced {pv} vs reference {rv} frames")
        else:
            print(f"  first divergence: frame {f}, codebook {c} (mlx={pv} ref={rv})")
    print(
        "  per-codebook agreement: "
        + " ".join(f"cb{c}={a:.2f}" for c, a in enumerate(per_cb))
    )

    audio_metrics = None
    if decode_audio:
        wav = model.decode_audio(mx.array(produced.astype(np.int32)))
        wav = np.asarray(wav).astype(np.float32)
        ref_wav_path = ref_dir / item["wav"]
        if ref_wav_path.exists():
            import wave

            with wave.open(str(ref_wav_path), "rb") as f:
                raw = f.readframes(f.getnframes())
                sw = f.getsampwidth()
            if sw == 2:
                ref_wav = (
                    np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
                )
            else:
                ref_wav = np.frombuffer(raw, dtype=np.float32)
            m = min(len(wav), len(ref_wav))
            if m > 0:
                diff = wav[:m] - ref_wav[:m]
                audio_metrics = {
                    "max_abs": float(np.max(np.abs(diff))),
                    "rms": float(np.sqrt(np.mean(diff**2))),
                    "len_mlx": len(wav),
                    "len_ref": len(ref_wav),
                }
                print(
                    f"  audio: max_abs={audio_metrics['max_abs']:.4f} "
                    f"rms={audio_metrics['rms']:.4f} "
                    f"(len mlx={len(wav)} ref={len(ref_wav)})"
                )

    return {
        "index": item["index"],
        "exact": exact,
        "total": total,
        "first_divergence": div,
        "per_codebook": per_cb,
        "audio": audio_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="/Volumes/DATA/zonos2-mlx",
        help="Converted MLX model directory (config.json + model.safetensors).",
    )
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--decode-audio", action="store_true")
    args = ap.parse_args()

    ref_dir = Path(args.reference)
    with open(ref_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"Loading MLX model from {args.model} ...")
    model = Model.from_local(args.model)

    results = [
        check_item(model, ref_dir, item, args.decode_audio)
        for item in manifest["items"]
    ]

    print("\n========== PARITY SUMMARY ==========")
    verdict_parts = []
    all_exact = True
    for r in results:
        tag = f"item{r['index']} {r['exact']}/{r['total']} frames exact"
        if r["first_divergence"] is not None:
            all_exact = False
            f, c, pv, rv = r["first_divergence"]
            if c != -1:
                tag += f" (first div frame {f} cb{c})"
            else:
                tag += f" (len {pv} vs {rv})"
        verdict_parts.append(tag)
    print("PARITY: " + "; ".join(verdict_parts))
    print("RESULT:", "PASS (all frames exact)" if all_exact else "FAIL (see above)")


if __name__ == "__main__":
    main()
