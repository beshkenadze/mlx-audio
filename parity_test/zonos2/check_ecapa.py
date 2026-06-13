"""ECAPA-TDNN voice-encoder parity check: MLX port vs the real torch checkpoint.

Loads the torch reference ``EcapaTdnnSpeakerEncoder`` (the architecture actually
shipped in ``marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B``, despite the "Qwen3"
name), converts the HF weights with ``convert_voice_embedder``, loads the MLX
``EcapaTdnnBackbone``, feeds the SAME reference waveform through both, and reports
cosine similarity + max abs diff between the two 2048-D x-vectors.

Run (fully local / CPU / tiny model):
    uv run --no-sync --with torch --with transformers --with torchaudio \
        --project /Volumes/DATA/mlx-audio python parity_test/zonos2/check_ecapa.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
_REF_WAV_CANDIDATES = [
    "/Volumes/DATA/mlx-audio-zonos2/parity_test/zonos2/reference/item_0.wav",
]
_TARGET_SR = 24_000


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV to mono float32 in [-1, 1] using the stdlib (no torchcodec)."""
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise ValueError(f"only 16-bit PCM supported, got sampwidth={sampwidth}")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data, sr


def _load_waveform() -> np.ndarray:
    """Return a fixed mono float32 24 kHz test waveform (reference wav or sine)."""
    for cand in _REF_WAV_CANDIDATES:
        if Path(cand).exists():
            import torch
            import torchaudio

            wav_np, sr = _read_wav(cand)
            wav = torch.from_numpy(wav_np)
            if sr != _TARGET_SR:
                wav = torchaudio.functional.resample(wav, sr, _TARGET_SR)
            print(
                f"  using reference wav: {cand} "
                f"({wav.shape[0]} samples @ {_TARGET_SR} Hz)"
            )
            return wav.numpy().astype(np.float32)
    # Fallback synthetic sine (3 s) — deterministic.
    n = _TARGET_SR * 3
    t = np.arange(n, dtype=np.float32) / _TARGET_SR
    wav = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    print(f"  using synthetic sine ({n} samples @ {_TARGET_SR} Hz)")
    return wav


def _torch_reference(wav: np.ndarray) -> np.ndarray:
    """Reference embedding [2048] from the HF torch model + its feature extractor."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval().to("cpu")
    fe = AutoFeatureExtractor.from_pretrained(REPO, trust_remote_code=True)
    feats = fe(wav, sampling_rate=_TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        out = model(input_values=feats["input_values"].float())
    emb = out.last_hidden_state.squeeze(0).float().numpy()
    return emb


def _mlx_port(wav: np.ndarray) -> np.ndarray:
    """MLX embedding [2048]: convert HF weights, load backbone, run frontend."""
    from mlx.utils import tree_unflatten

    from mlx_audio.tts.models.zonos2.config import ZONOS2Config
    from mlx_audio.tts.models.zonos2.convert import convert_voice_embedder
    from mlx_audio.tts.models.zonos2.speaker_encoder import Qwen3VoiceEmbedding

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = convert_voice_embedder(REPO, tmp, dtype="float32")
        weights = mx.load(str(Path(out_dir) / "model.safetensors"))

        encoder = Qwen3VoiceEmbedding(ZONOS2Config())
        encoder.backbone.update(tree_unflatten(list(weights.items())))
        encoder.eval()
        mx.eval(encoder.parameters())

        emb = encoder(mx.array(wav), sample_rate=_TARGET_SR)
        mx.eval(emb)
    return np.array(emb.squeeze(0), dtype=np.float32)


def main() -> None:
    print("ECAPA-TDNN voice-encoder parity: MLX port vs torch reference")
    wav = _load_waveform()

    print("Running torch reference...")
    ref = _torch_reference(wav)
    print("Running MLX port...")
    mlx_emb = _mlx_port(wav)

    assert ref.shape == (2048,), f"reference shape {ref.shape}"
    assert mlx_emb.shape == (2048,), f"mlx shape {mlx_emb.shape}"

    cos = float(
        np.dot(ref, mlx_emb) / (np.linalg.norm(ref) * np.linalg.norm(mlx_emb) + 1e-12)
    )
    maxabs = float(np.max(np.abs(ref - mlx_emb)))
    print()
    print(f"  reference norm = {np.linalg.norm(ref):.4f}")
    print(f"  mlx       norm = {np.linalg.norm(mlx_emb):.4f}")
    print(f"ECAPA: cosine={cos:.6f} maxabs={maxabs:.6e}")
    print("PASS" if cos >= 0.999 else "FAIL (cosine < 0.999)")


if __name__ == "__main__":
    main()
