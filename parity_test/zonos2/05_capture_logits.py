#!/usr/bin/env python
"""Capture ZONOS2 CUDA pre-sampling logits for teacher-forced parity (pc.lan).

Run inside the zonos2-parity torch env (RTX 4090). For items 0 & 1 this loads
the offline ``TTSLLM(model_path="Zyphra/ZONOS2")`` and runs the SAME greedy
generation used to build the reference fixtures, while recording the
**pre-sampling, post-softcap logits at every decode step** (shape per step
``[n_codebooks, audio_vocab]``).

Hook: the engine samples OUTSIDE the CUDA graph via
``zonos2.engine.sample.TTSSampler.sample`` (which calls ``sample_tts`` on
``logits: (B, n_codebooks, vocab)``). We monkeypatch that method to stash
``logits[0].detach().float().cpu().numpy()`` per step, then delegate to the
original. The recorded logits are already soft-capped (the model head applies
``loss_softcap`` before returning), matching the MLX port's post-softcap logits.

Sanity: the recorded greedy tokens (argmax of the captured logits) must equal
the reference ``audio_tokens`` for each item.

Output: ``ref_logits_item{0,1}.npz`` with
  * ``logits``  [n_frames, n_codebooks, audio_vocab] float32
  * ``tokens``  [n_frames, n_codebooks] int32 (argmax sanity copy)

Usage (inside tmux on pc.lan):
    cd /mnt/d/Projects/zonos2-parity/ZONOS2
    HF_HOME=/mnt/d/.cache/hf \
        /home/linuxbrew/.linuxbrew/bin/uv run --project . python \
        /mnt/d/Projects/zonos2-parity/05_capture_logits.py \
        --out /mnt/d/Projects/zonos2-parity/reflogits
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

# Greedy params identical to 04_make_reference.py (the reference gate).
GREEDY = dict(
    temperature=1.0,
    topk=1,
    top_p=0.0,
    min_p=0.0,
    repetition_penalty=1.0,
    max_tokens=128,
    ignore_eos=False,
    seed=0,
)

PROMPTS = [
    "Hello world, this is a parity test.",
    "The quick brown fox jumps over the lazy dog.",
]

MODEL = "Zyphra/ZONOS2"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/d/Projects/zonos2-parity/reflogits")
    ap.add_argument(
        "--items",
        default="0,1",
        help="Comma-separated reference item indices to capture (default 0,1).",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = [int(x) for x in args.items.split(",") if x.strip() != ""]

    import torch  # noqa: F401  (imported for side effects / availability)
    from zonos2.engine.sample import TTSSampler
    from zonos2.message import TTSSamplingParams
    from zonos2.tts import TTSLLM

    # Per-step logit buffer; the engine processes one request at a time here
    # (single prompt per generate call), so logits[0] is THIS request's frame.
    captured: List[np.ndarray] = []

    original_sample = TTSSampler.sample

    def sample_capture(self, logits, args_):  # noqa: ANN001
        # logits: (B, n_codebooks, vocab); B == 1 for our single-prompt runs.
        captured.append(logits[0].detach().float().cpu().numpy())
        return original_sample(self, logits, args_)

    TTSSampler.sample = sample_capture

    print(f"Loading TTSLLM({MODEL}) ...", flush=True)
    tts = TTSLLM(model_path=MODEL, decode_audio=False)

    for i in items:
        text = PROMPTS[i]
        captured.clear()
        sp = TTSSamplingParams(**GREEDY)
        res = tts.generate([text], sp)[0]
        audio_tokens = np.asarray(res["audio_tokens"], dtype=np.int32)  # [F, C]
        n_frames, n_cb = audio_tokens.shape

        # One capture entry per decode step. Greedy stops via the max_tokens cap
        # (ignore_eos=False, but these prompts never emit EOA within 128 frames),
        # so the capture count must equal n_frames EXACTLY -- any extra leading
        # (warmup) or trailing capture, or an early EOA stop, would misalign the
        # array, so we require an exact count rather than slicing from one end.
        assert len(captured) == n_frames, (
            f"item{i}: captured {len(captured)} sample() calls != {n_frames} "
            f"decode frames -- the hook is misaligned (extra/missing sample call)"
        )
        logits = np.stack(captured, axis=0).astype(np.float32)  # [F, C, V]
        assert (
            logits.shape[1] == n_cb
        ), f"item{i}: captured C={logits.shape[1]} != n_codebooks={n_cb}"

        argmax_tokens = logits.argmax(axis=-1).astype(np.int32)  # [F, C]
        # Sanity: greedy argmax of the captured logits must reproduce the sampled
        # tokens. With topk=1 + temperature=1.0, sample_tts collapses to argmax,
        # so any disagreement beyond a handful of exact GPU-vs-CPU ties means the
        # hook captured the wrong tensor / wrong frame. A few ties are expected
        # (GPU multinomial vs CPU-float32 argmax on equal logits); a large rate
        # is a capture bug, so gate on a tight tolerance.
        mismatch = int((argmax_tokens != audio_tokens).sum())
        agree = 100.0 * (argmax_tokens == audio_tokens).mean()
        max_allowed = max(5, int(0.01 * argmax_tokens.size))  # <=1% (or 5) ties
        assert mismatch <= max_allowed, (
            f"item{i}: argmax(captured) disagrees with sampled tokens at "
            f"{mismatch}/{argmax_tokens.size} positions (> {max_allowed}); the "
            f"capture hook is recording the wrong logits"
        )
        print(
            f"[item{i}] frames={n_frames} cb={n_cb} vocab={logits.shape[2]} "
            f"argmax-vs-ref mismatches={mismatch} agree={agree:.2f}%",
            flush=True,
        )

        out_path = os.path.join(args.out, f"ref_logits_item{i}.npz")
        np.savez(
            out_path,
            logits=logits,
            tokens=audio_tokens,
            argmax_tokens=argmax_tokens,
        )
        print(f"  saved {out_path} (logits {logits.shape} {logits.dtype})", flush=True)

    TTSSampler.sample = original_sample
    print("DONE capture ->", args.out, flush=True)


if __name__ == "__main__":
    main()
