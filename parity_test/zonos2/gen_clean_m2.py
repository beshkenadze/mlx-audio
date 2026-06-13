"""STEP B + C: clean MLX audio (plain + voice-clone) from the captured clean prompt.

Two things, one watchdog'd 8B load:

  STEP B — feed the CUDA-captured CLEAN-conditioned prompt (``clean_prompt_item{i}.npy``,
  which carries the high-SNR / good-loudness / full-bandwidth quality tokens) to the
  MLX backbone:
    * plain        -> ``mlx_clean_item{i}.wav``
    * voice-clone  -> ``mlx_clean_clone_item{i}.wav`` (AmericanMale ECAPA x-vector
      injected at the canonical speaker slot; the clean prompt carries the quality
      conditioning, the embedding carries the voice).

  STEP C — verify the MLX text->prompt builder is self-sufficient: build
  ``Model.build_prompt_ids(text, quality_buckets=CLEAN_QB)`` and assert it matches the
  captured ``clean_prompt_item{i}.npy`` BYTE-FOR-BYTE. If it matches, MLX text->speech
  needs no CUDA capture.

Run (Mac, one job at a time, hard 20 GB RSS watchdog):
  cd <worktree>
  PYTHONPATH=$PWD uv run --no-sync --project /Volumes/DATA/mlx-audio \
      python parity_test/zonos2/gen_clean_m2.py
"""

from __future__ import annotations

import os
import resource
import threading
import time
import wave
from pathlib import Path

import mlx.core as mx

# --- hard memory cap (<= 20 GB shared Mac); the 8B clean gen peaks ~19 GB. ---
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

SPK_WEIGHTS = "/Volumes/DATA/zonos2-mlx-spk"
BASE_WEIGHTS = "/Volumes/DATA/zonos2-mlx"
SPK_SAFETENSORS = "/Volumes/DATA/zonos2-spk/speaker.safetensors"
COORD = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2")
CLEAN_DIR = COORD / "clean"
EMB_NPY = COORD / "spk_emb_americanmale.npy"
SR = 44100

# Texts + the clean buckets MUST match 08_capture_clean.py (the CUDA capture).
TEXTS = [
    "Hello world, this is a parity test.",
    "The quick brown fox jumps over the lazy dog.",
    "Numbers like 42 and dates such as June 13th should normalize cleanly.",
]
CLEAN_QB = {
    "lufs": 7,
    "estimated_snr": 11,
    "max_pause": 0,
    "estimated_bandlimit_hz": 7,
    "leading_silence_s": 0,
    "trailing_silence_s": 0,
}

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


def edge_silence(wav: np.ndarray, thresh: float = 0.01) -> tuple[float, float]:
    if wav.size == 0:
        return 0.0, 0.0
    above = np.abs(wav) >= thresh
    if not above.any():
        return wav.size / SR, wav.size / SR
    first = int(np.argmax(above))
    last = int(len(above) - 1 - np.argmax(above[::-1]))
    return first / SR, (len(wav) - 1 - last) / SR


def stats(wav: np.ndarray) -> str:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0
    lead, trail = edge_silence(wav)
    return (
        f"dur={wav.size / SR:.2f}s peak={peak:.3f} rms={rms:.4f} "
        f"lead_sil={lead * 1000:.0f}ms trail_sil={trail * 1000:.0f}ms"
    )


def load_model() -> Model:
    if Path(SPK_WEIGHTS, "model.safetensors").exists():
        print(f"loading voice-cloning checkpoint {SPK_WEIGHTS} ...", flush=True)
        return Model.from_local(SPK_WEIGHTS)
    print(f"loading base {BASE_WEIGHTS} + speaker {SPK_SAFETENSORS} ...", flush=True)
    return Model.from_local(BASE_WEIGHTS, speaker_weights=SPK_SAFETENSORS)


def verify_step_c(model: Model) -> None:
    """STEP C: MLX-built clean prompt must match the captured CUDA prompt exactly."""
    print("\n=== STEP C: MLX build_prompt_ids vs captured CUDA clean prompt ===")
    all_ok = True
    for i, text in enumerate(TEXTS):
        cap_path = CLEAN_DIR / f"clean_prompt_item{i}.npy"
        if not cap_path.exists():
            print(f"  item{i}: SKIP (no captured prompt at {cap_path})", flush=True)
            all_ok = False
            continue
        captured = np.load(cap_path).astype(np.int32)
        built = np.asarray(
            model.build_prompt_ids(text, quality_buckets=CLEAN_QB)
        ).astype(np.int32)
        match = built.shape == captured.shape and np.array_equal(built, captured)
        all_ok = all_ok and match
        tag = "MATCH" if match else "MISMATCH"
        print(
            f"  item{i}: {tag} built={built.shape} captured={captured.shape}",
            flush=True,
        )
        if not match:
            n = min(built.shape[0], captured.shape[0])
            diffs = [
                (r, int(built[r, -1]), int(captured[r, -1]))
                for r in range(n)
                if not np.array_equal(built[r], captured[r])
            ][:8]
            print(
                f"    first text-col diffs (row, built, captured): {diffs}", flush=True
            )
    print(
        "STEP C RESULT: "
        + ("SELF-SUFFICIENT (MLX text->prompt matches CUDA)" if all_ok else "MISMATCH"),
        flush=True,
    )


def main() -> None:
    emb = mx.array(np.load(EMB_NPY).astype(np.float32))
    print(
        f"speaker embedding {tuple(emb.shape)} norm={float(mx.linalg.norm(emb)):.3f}",
        flush=True,
    )

    model = load_model()

    # STEP C first (cheap; no generation) so we know whether MLX is self-sufficient.
    verify_step_c(model)

    # STEP B: plain clean from the captured clean prompt.
    print("\n=== STEP B: plain clean MLX audio ===", flush=True)
    for i in range(len(TEXTS)):
        prompt = mx.array(
            np.load(CLEAN_DIR / f"clean_prompt_item{i}.npy").astype(np.int32)
        )
        mx.random.seed(42)
        codes, waveform = model.generate(
            "(prebuilt clean prompt)",
            prompt_ids=prompt,
            max_frames=MAX_FRAMES,
            stop_on_eoa=True,
            decode_audio=True,
            trim_leading_silence=True,
            **SAMPLER,
        )
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        out = COORD / f"mlx_clean_item{i}.wav"
        save_wav(out, wav)
        n_frames = int(np.asarray(codes).shape[0])
        print(
            f"  mlx_clean item{i}: frames={n_frames} {stats(wav)} -> {out}", flush=True
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    # STEP B: voice-clone clean (clean prompt + AmericanMale embedding).
    print("\n=== STEP B: voice-clone clean MLX audio ===", flush=True)
    for i in range(len(TEXTS)):
        prompt = mx.array(
            np.load(CLEAN_DIR / f"clean_prompt_item{i}.npy").astype(np.int32)
        )
        mx.random.seed(42)
        codes, waveform = model.generate(
            "(prebuilt clean prompt)",
            prompt_ids=prompt,
            speaker_embedding=emb,
            max_frames=MAX_FRAMES,
            stop_on_eoa=True,
            decode_audio=True,
            trim_leading_silence=True,
            **SAMPLER,
        )
        wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
        out = COORD / f"mlx_clean_clone_item{i}.wav"
        save_wav(out, wav)
        n_frames = int(np.asarray(codes).shape[0])
        print(
            f"  mlx_clean_clone item{i}: frames={n_frames} {stats(wav)} -> {out}",
            flush=True,
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    print(f"\npeak MLX mem: {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
