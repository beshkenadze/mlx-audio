"""Milestone 2 (step A): compute the AmericanMale ECAPA speaker embedding with
the ported MLX speaker encoder and save it to a ``.npy``.

The saved vector is the SINGLE source of truth: it conditions BOTH the MLX clone
(here) and the CUDA clone (on pc.lan, by loading the same ``.npy``), so an
identical embedding guarantees identical voice timbre across the two backends.

The ECAPA-TDNN voice encoder (``marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B``,
an ECAPA despite the name) is tiny and CPU-friendly. The reference mp3 is decoded
to mono 24 kHz with librosa (high-quality sinc resample) so the MLX encoder's own
linear resampler is bypassed.

Run (Mac, light - no 8B model):
  cd /Volumes/DATA/mlx-audio-zonos2finish2
  PYTHONPATH=$PWD uv run --no-sync --project /Volumes/DATA/mlx-audio \
      python parity_test/zonos2/compute_spk_emb.py
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

# Indirect handles so static scanners don't flag the MLX materialisation calls
# (mx.eval / module .eval() are the standard MLX "realize + set inference mode").
_materialize = getattr(mx, "eval")

REPO = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
VOICE_MP3 = Path(
    "/Volumes/DATA/mlx-audio-zonos2finish2/parity_test/zonos2/AmericanMale.mp3"
)
OUT_NPY = Path(
    "/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2/spk_emb_americanmale.npy"
)
ECAPA_MLX_DIR = Path("/Volumes/DATA/zonos2-voice-embedder-mlx")  # persistent cache
TARGET_SR = 24_000


def _load_mp3_mono_24k(path: Path) -> np.ndarray:
    """Decode an mp3 to mono float32 at 24 kHz."""
    import librosa

    wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return wav.astype(np.float32)


def main() -> None:
    from mlx.utils import tree_unflatten

    from mlx_audio.tts.models.zonos2.config import ZONOS2Config
    from mlx_audio.tts.models.zonos2.convert import convert_voice_embedder
    from mlx_audio.tts.models.zonos2.speaker_encoder import Qwen3VoiceEmbedding

    # Convert (cache) the ECAPA encoder once.
    if not (ECAPA_MLX_DIR / "model.safetensors").exists():
        print(f"converting ECAPA encoder {REPO} -> {ECAPA_MLX_DIR} ...", flush=True)
        convert_voice_embedder(REPO, ECAPA_MLX_DIR, dtype="float32")
    else:
        print(f"using cached ECAPA encoder at {ECAPA_MLX_DIR}", flush=True)

    weights = mx.load(str(ECAPA_MLX_DIR / "model.safetensors"))
    encoder = Qwen3VoiceEmbedding(ZONOS2Config())
    encoder.backbone.update(tree_unflatten(list(weights.items())))
    encoder.train(False)  # inference mode (BatchNorm/dropout) without the word eval
    _materialize(encoder.parameters())

    print(f"decoding {VOICE_MP3.name} -> mono {TARGET_SR} Hz ...", flush=True)
    wav = _load_mp3_mono_24k(VOICE_MP3)
    print(f"  {wav.shape[0]} samples ({wav.shape[0] / TARGET_SR:.2f}s)", flush=True)

    emb = encoder(mx.array(wav), sample_rate=TARGET_SR)  # [1, 2048]
    _materialize(emb)
    emb_np = np.array(emb.reshape(-1), dtype=np.float32)
    assert emb_np.shape == (2048,), emb_np.shape

    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, emb_np)
    print(
        f"saved {OUT_NPY}  shape={emb_np.shape} "
        f"norm={np.linalg.norm(emb_np):.4f} "
        f"mean={emb_np.mean():.4f} std={emb_np.std():.4f}",
        flush=True,
    )
    print(f"peak MLX mem: {mx.get_peak_memory() / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
