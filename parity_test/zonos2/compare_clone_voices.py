"""Objective voice-match check: re-embed the generated clones with the ECAPA
encoder and report speaker cosine similarity vs the AmericanMale reference and
between the MLX and CUDA clones.

If the clones truly carry the AmericanMale timbre, their ECAPA embeddings should
sit much closer to the reference than the un-conditioned Milestone-1 audio, and
the MLX vs CUDA clones (same conditioning vector) should be close to each other.

Run (Mac, light):
  cd /Volumes/DATA/mlx-audio-zonos2finish2
  PYTHONPATH=$PWD uv run --no-sync --project /Volumes/DATA/mlx-audio \
      python parity_test/zonos2/compare_clone_voices.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

_materialize = getattr(mx, "eval")

ECAPA_MLX_DIR = Path("/Volumes/DATA/zonos2-voice-embedder-mlx")
OUT_DIR = Path("/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2")
REF_MP3 = Path(
    "/Volumes/DATA/mlx-audio-zonos2finish2/parity_test/zonos2/AmericanMale.mp3"
)
TARGET_SR = 24_000


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, sr


def _to_24k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return wav
    import librosa

    return librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    from mlx.utils import tree_unflatten

    from mlx_audio.tts.models.zonos2.config import ZONOS2Config
    from mlx_audio.tts.models.zonos2.speaker_encoder import Qwen3VoiceEmbedding

    weights = mx.load(str(ECAPA_MLX_DIR / "model.safetensors"))
    enc = Qwen3VoiceEmbedding(ZONOS2Config())
    enc.backbone.update(tree_unflatten(list(weights.items())))
    enc.train(False)
    _materialize(enc.parameters())

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
        wav24 = _to_24k(wav, sr)
        e = enc(mx.array(wav24), sample_rate=TARGET_SR)
        _materialize(e)
        return np.array(e.reshape(-1), dtype=np.float32)

    import librosa

    ref_wav, _ = librosa.load(str(REF_MP3), sr=TARGET_SR, mono=True)
    ref_emb = embed(ref_wav.astype(np.float32), TARGET_SR)

    print(f"{'item':<6}{'mlx~ref':>10}{'cuda~ref':>10}{'mlx~cuda':>10}{'m1~ref':>10}")
    for idx in [0, 1, 2]:
        embs = {}
        for tag in ("mlx_clone", "cuda_clone", "mlx_fix"):
            p = OUT_DIR / f"{tag}_item{idx}.wav"
            if not p.exists():
                embs[tag] = None
                continue
            wav, sr = _read_wav_mono(p)
            embs[tag] = embed(wav, sr) if wav.size > 2400 else None

        def c(a, b):
            if embs.get(a) is None or (b == "ref" and ref_emb is None):
                return float("nan")
            other = ref_emb if b == "ref" else embs.get(b)
            return _cos(embs[a], other) if other is not None else float("nan")

        print(
            f"{idx:<6}"
            f"{c('mlx_clone','ref'):>10.3f}"
            f"{c('cuda_clone','ref'):>10.3f}"
            f"{c('mlx_clone','cuda_clone'):>10.3f}"
            f"{c('mlx_fix','ref'):>10.3f}"
        )
    print(
        "\nHigher mlx~ref / cuda~ref than m1~ref => the clone carries the reference "
        "voice; high mlx~cuda => the two backends agree on timbre."
    )


if __name__ == "__main__":
    main()
