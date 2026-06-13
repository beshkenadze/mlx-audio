"""Milestone 2 (step B): generate MLX audio cloned to the AmericanMale voice.

Loads the voice-cloning checkpoint (``zonos2-mlx-spk``, which carries the merged
``speaker_lda_projection`` / ``speaker_projection`` tensors), conditions on the
SAME ECAPA embedding saved by ``compute_spk_emb.py`` (so the CUDA run can load the
identical vector), and synthesises items 0,1,2. ``Model.generate`` wraps the
prompt with the canonical speaker slot + clean-background marker and injects the
projected embedding at the speaker token position (mirrors upstream
``_with_speaker_frames`` / ``_forward_model``).

Run (Mac, watchdog'd, one job at a time):
  cd /Volumes/DATA/mlx-audio-zonos2finish2
  PYTHONPATH=$PWD uv run --no-sync --project /Volumes/DATA/mlx-audio \
      python parity_test/zonos2/gen_clone_m2.py
"""

from __future__ import annotations

import os
import resource
import threading
import time
import wave
from pathlib import Path

import mlx.core as mx

mx.set_memory_limit(int(18e9))
mx.set_cache_limit(int(1e9))
try:
    mx.set_wired_limit(int(18e9))
except Exception:
    pass


def _watchdog() -> None:
    while True:
        time.sleep(2)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 20 * 1024**3:
            os._exit(137)


threading.Thread(target=_watchdog, daemon=True).start()

import numpy as np  # noqa: E402

from mlx_audio.tts.models.zonos2.zonos2 import Model  # noqa: E402

# Speaker checkpoint (base 8B + merged speaker projections). Falls back to the
# base checkpoint + the extracted speaker.safetensors if the merged dir is absent.
SPK_WEIGHTS = "/Volumes/DATA/zonos2-mlx-spk"
BASE_WEIGHTS = "/Volumes/DATA/zonos2-mlx"
SPK_SAFETENSORS = "/Volumes/DATA/zonos2-spk/speaker.safetensors"
REF_DIR = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2/reference")
OUT_DIR = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2")
EMB_NPY = OUT_DIR / "spk_emb_americanmale.npy"
SR = 44100

SAMPLER = dict(
    temperature=1.15,
    top_k=106,
    top_p=0.0,
    min_p=0.18,
    repetition_penalty=1.2,
    repetition_window=50,
    repetition_codebooks=8,
)
MAX_FRAMES = int(os.environ.get("MAXF", "768"))


def save_wav(path: Path, wav: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _edge_silence(wav: np.ndarray, thresh: float = 0.01) -> tuple[float, float]:
    if wav.size == 0:
        return 0.0, 0.0
    above = np.abs(wav) >= thresh
    if not above.any():
        return wav.size / SR, wav.size / SR
    first = int(np.argmax(above))
    last = int(len(above) - 1 - np.argmax(above[::-1]))
    return first / SR, (len(wav) - 1 - last) / SR


def _load_model() -> Model:
    if Path(SPK_WEIGHTS, "model.safetensors").exists():
        print(f"loading voice-cloning checkpoint {SPK_WEIGHTS} ...", flush=True)
        return Model.from_local(SPK_WEIGHTS)
    print(f"loading base {BASE_WEIGHTS} + speaker {SPK_SAFETENSORS} ...", flush=True)
    return Model.from_local(BASE_WEIGHTS, speaker_weights=SPK_SAFETENSORS)


def main() -> None:
    emb = mx.array(np.load(EMB_NPY).astype(np.float32))
    print(
        f"speaker embedding {tuple(emb.shape)} norm={float(mx.linalg.norm(emb)):.3f}",
        flush=True,
    )

    model = _load_model()

    for idx in [0, 1, 2]:
        ref = np.load(REF_DIR / f"item_{idx}.npz")
        prompt_ids = mx.array(ref["prompt_ids"].astype(np.int32))
        mx.random.seed(42)
        print(f"generating clone item {idx} ...", flush=True)
        codes, waveform = model.generate(
            "(prebuilt prompt_ids)",
            prompt_ids=prompt_ids,
            speaker_embedding=emb,
            max_frames=MAX_FRAMES,
            stop_on_eoa=True,
            decode_audio=True,
            trim_leading_silence=True,
            **SAMPLER,
        )
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        out = OUT_DIR / f"mlx_clone_item{idx}.wav"
        save_wav(out, wav)
        dur = wav.size / SR
        lead, trail = _edge_silence(wav)
        n_frames = int(np.asarray(codes).shape[0])
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        print(
            f"  clone item{idx}: frames={n_frames} dur={dur:.2f}s "
            f"lead_sil={lead*1000:.0f}ms trail_sil={trail*1000:.0f}ms "
            f"peak={peak:.3f} -> {out}",
            flush=True,
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    print(f"peak MLX mem: {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
