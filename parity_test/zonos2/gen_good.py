"""Generate full-quality ZONOS2 MLX audio with the complete sampler.

Uses the upstream ``TTSSamplingParams`` defaults — temperature 1.15, top_k 106,
top_p 0.0 (disabled), min_p 0.18, repetition_penalty 1.2, window 50, the first 8
of 9 codebooks penalized — so the second half of each clip no longer collapses
into a jumble (repetition_penalty + min_p were the missing pieces).

Writes wavs to the shared coordinator dir and prints per-item stats incl. a
2nd-half-vs-1st-half unique-frame degradation check, comparing against the CUDA
"good" reference wavs.

Usage:
  cd /Volumes/DATA/mlx-audio/.claude/worktrees/agent-a1661a473b1aaafa4
  PYTHONPATH=$PWD uv run --no-sync --project /Volumes/DATA/mlx-audio \
      python parity_test/zonos2/gen_good.py
"""

import os
import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

# --- hard memory cap (<=20 GB shared Mac) ---
mx.set_memory_limit(int(18e9))
mx.set_cache_limit(int(1e9))

from mlx_audio.tts.models.zonos2.zonos2 import Model  # noqa: E402

MLX_WEIGHTS = "/Volumes/DATA/zonos2-mlx"
# Shared coordinator dir (stable paths the user listens to).
COORD = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2")
REF = COORD / "reference"
GOOD = COORD / "good"

# Upstream TTSSamplingParams defaults.
TEMP = float(os.environ.get("TEMP", "1.15"))
TOPK = int(os.environ.get("TOPK", "106"))
TOPP = float(os.environ.get("TOPP", "0.0"))
MINP = float(os.environ.get("MINP", "0.18"))
REP_PEN = float(os.environ.get("REP_PEN", "1.2"))
REP_WIN = int(os.environ.get("REP_WIN", "50"))
REP_CB = int(os.environ.get("REP_CB", "8"))
MAXF = int(os.environ.get("MAXF", "512"))


def save_wav(path, wav, sr=44100):
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_wav_rms(path):
    if not Path(path).exists():
        return None
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return float(np.sqrt(np.mean(a**2))) if a.size else 0.0


def unique_ratio(frames_np):
    """Fraction of distinct frames (proxy for how 'alive' the codes are)."""
    if len(frames_np) == 0:
        return 0.0
    return len({tuple(r) for r in frames_np.tolist()}) / len(frames_np)


def main():
    indices = [0, 1, 2]
    mx.random.seed(42)
    print(
        f"[full sampler] temp={TEMP} topk={TOPK} top_p={TOPP} min_p={MINP} "
        f"rep_pen={REP_PEN} win={REP_WIN} cb={REP_CB} maxf={MAXF}",
        flush=True,
    )
    print(f"loading MLX model from {MLX_WEIGHTS} ...", flush=True)
    model = Model.from_local(MLX_WEIGHTS)

    for idx in indices:
        ref = np.load(REF / f"item_{idx}.npz")
        prompt_ids = mx.array(ref["prompt_ids"].astype(np.int32))
        print(f"generating item {idx} ...", flush=True)
        codes, waveform = model.generate(
            "(prebuilt prompt_ids)",
            prompt_ids=prompt_ids,
            max_frames=MAXF,
            temperature=TEMP,
            top_k=TOPK,
            top_p=TOPP,
            min_p=MINP,
            repetition_penalty=REP_PEN,
            repetition_window=REP_WIN,
            repetition_codebooks=REP_CB,
            decode_audio=True,
        )
        out = COORD / f"mlx_good_item{idx}.wav"
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        save_wav(out, wav)

        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        rms = float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0
        sil = float(np.mean(np.abs(wav) < 0.01)) if wav.size else 1.0

        codes_np = np.array(codes)
        n = len(codes_np)
        uniq = len({tuple(r) for r in codes_np.tolist()})
        half = n // 2
        first_ratio = unique_ratio(codes_np[:half])
        second_ratio = unique_ratio(codes_np[half:])
        degr = (second_ratio / first_ratio) if first_ratio > 0 else 0.0

        good_rms = load_wav_rms(GOOD / f"good_item{idx}.wav")
        good_str = f"{good_rms:.4f}" if good_rms is not None else "n/a"
        print(
            f"  item{idx}: codes={tuple(codes_np.shape)} unique_frames={uniq} "
            f"peak={peak:.3f} rms={rms:.4f} silence={sil*100:.1f}%\n"
            f"           uniq_ratio 1st={first_ratio:.3f} 2nd={second_ratio:.3f} "
            f"2nd/1st={degr:.3f} (>=~0.8 = no late collapse)\n"
            f"           CUDA good_item{idx} rms={good_str}  -> {out}",
            flush=True,
        )
        print(
            f"           RSS now: {mx.get_peak_memory() / 1e9:.2f} GB peak", flush=True
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    print(f"peak MLX mem: {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
