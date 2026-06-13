"""Milestone 1: regenerate items 0,1,2 WITHOUT the end-cut + with the leading
pause trimmed, using the full-quality sampler.

The captured reference items were produced with ``max_tokens=128`` and never
reached the end-of-audio token (every ``eos_frame`` is null), so the reference
itself is cut off mid-sentence. Here we feed the EXACT same captured prompts but
let the model finish naturally: ``stop_on_eoa=True`` with a generous
``max_frames``; generation halts at the model's own end-of-audio token and the
delay tail is flushed and trimmed at the aligned ``eos_frame``. The decoded
waveform additionally has its trained ~0.2 s leading-silence prefix stripped.

Run (Mac, watchdog'd, one job at a time):
  cd /Volumes/DATA/mlx-audio-zonos2finish2
  PYTHONPATH=$PWD uv run --no-sync --project /Volumes/DATA/mlx-audio \
      python parity_test/zonos2/gen_fix_m1.py
"""

from __future__ import annotations

import os
import resource
import sys
import threading
import time
import wave
from pathlib import Path

import mlx.core as mx

# --- hard memory cap (shared 32 GB Mac; keep the 8B generate < 20 GB) ---
mx.set_memory_limit(int(18e9))
mx.set_cache_limit(int(1e9))
try:
    mx.set_wired_limit(int(18e9))
except Exception:
    pass


def _watchdog() -> None:
    """Hard-exit if RSS exceeds 20 GB (ru_maxrss is bytes on macOS)."""
    while True:
        time.sleep(2)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 20 * 1024**3:
            os._exit(137)


threading.Thread(target=_watchdog, daemon=True).start()

import numpy as np  # noqa: E402

from mlx_audio.tts.models.zonos2.zonos2 import Model  # noqa: E402

MLX_WEIGHTS = "/Volumes/DATA/zonos2-mlx"
REF_DIR = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2/reference")
OUT_DIR = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2")
SR = 44100

# Upstream TTSSamplingParams defaults (the clean, full-quality sampler).
SAMPLER = dict(
    temperature=1.15,
    top_k=106,
    top_p=0.0,
    min_p=0.18,
    repetition_penalty=1.2,
    repetition_window=50,
    repetition_codebooks=8,
)
MAX_FRAMES = int(os.environ.get("MAXF", "768"))  # ~8.9 s ceiling; sentences are short


def save_wav(path: Path, wav: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _edge_silence(wav: np.ndarray, thresh: float = 0.01) -> tuple[float, float]:
    """Return (leading, trailing) silence durations in seconds."""
    if wav.size == 0:
        return 0.0, 0.0
    above = np.abs(wav) >= thresh
    if not above.any():
        return wav.size / SR, wav.size / SR
    first = int(np.argmax(above))
    last = int(len(above) - 1 - np.argmax(above[::-1]))
    return first / SR, (len(wav) - 1 - last) / SR


def main() -> None:
    indices = [0, 1, 2]
    mx.random.seed(42)
    print(f"loading MLX model from {MLX_WEIGHTS} (maxf={MAX_FRAMES}) ...", flush=True)
    model = Model.from_local(MLX_WEIGHTS)

    for idx in indices:
        ref = np.load(REF_DIR / f"item_{idx}.npz")
        prompt_ids = mx.array(ref["prompt_ids"].astype(np.int32))
        mx.random.seed(42)  # identical seed per item for reproducibility
        print(f"generating item {idx} (stop_on_eoa) ...", flush=True)
        codes, waveform = model.generate(
            "(prebuilt prompt_ids)",
            prompt_ids=prompt_ids,
            max_frames=MAX_FRAMES,
            stop_on_eoa=True,
            decode_audio=True,
            trim_leading_silence=True,
            **SAMPLER,
        )
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        out = OUT_DIR / f"mlx_fix_item{idx}.wav"
        save_wav(out, wav)
        dur = wav.size / SR
        lead, trail = _edge_silence(wav)
        n_frames = int(np.asarray(codes).shape[0])
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        print(
            f"  item{idx}: frames={n_frames} dur={dur:.2f}s "
            f"lead_sil={lead*1000:.0f}ms trail_sil={trail*1000:.0f}ms "
            f"peak={peak:.3f} -> {out}",
            flush=True,
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    print(f"peak MLX mem: {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
