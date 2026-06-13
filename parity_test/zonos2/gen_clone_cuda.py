"""Milestone 2 (step C): CUDA reference clone with the SAME ECAPA embedding.

Run on pc.lan in the zonos2-parity env. Loads the embedding produced by the MLX
``compute_spk_emb.py`` (rsynced to pc.lan) and drives the offline ``TTSLLM`` with
``speaker_embedding`` set on every request, so the CUDA backbone conditions on the
identical x-vector the MLX clone used. An identical embedding => identical voice
timbre.

The offline ``TTSLLM`` API does not expose ``speaker_embedding`` on ``generate``,
so we override ``offline_receive_msg`` to attach it to each ``TTSUserMsg`` (the
scheduler then prepends the canonical speaker slot + clean-background marker and
injects the embedding at position 0 — exactly like the MLX port). ``accurate_mode``
is False to match the MLX prompt layout (no accurate-mode marker).

Run:
  cd /mnt/d/Projects/zonos2-parity
  /home/linuxbrew/.linuxbrew/bin/uv run --project /mnt/d/Projects/nemo-test \
      python gen_clone_cuda.py    # or whatever env has zonos2 + torch + CUDA
"""

import os
import wave

import numpy as np
import torch
from zonos2.message import TTSSamplingParams, TTSUserMsg
from zonos2.tts import TTSLLM

OUT = "/mnt/d/Projects/zonos2-parity/clone"
EMB_NPY = "/mnt/d/Projects/zonos2-parity/spk_emb_americanmale.npy"
TEXTS = [
    "Hello world, this is a parity test.",
    "The quick brown fox jumps over the lazy dog.",
    "Numbers like 42 and dates such as June 13th should normalize cleanly.",
]


class ClonedTTSLLM(TTSLLM):
    """TTSLLM that injects a fixed speaker embedding into every request."""

    def __init__(self, *args, speaker_embedding=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._clone_emb = speaker_embedding

    def offline_receive_msg(self, blocking: bool = False):
        msgs = super().offline_receive_msg(blocking=blocking)
        for msg in msgs:
            if isinstance(msg, TTSUserMsg) and self._clone_emb is not None:
                msg.speaker_embedding = self._clone_emb
                msg.speaker_token_position = 0
                msg.clean_speaker_background = True
                msg.accurate_mode = False
        return msgs


def main():
    os.makedirs(OUT, exist_ok=True)
    emb = torch.from_numpy(np.load(EMB_NPY).astype(np.float32))
    print(
        f"speaker embedding shape={tuple(emb.shape)} norm={emb.norm().item():.3f}",
        flush=True,
    )

    tts = ClonedTTSLLM(
        model_path="Zyphra/ZONOS2", decode_audio=True, speaker_embedding=emb
    )

    for i, text in enumerate(TEXTS):
        sp = TTSSamplingParams(seed=42)
        res = tts.generate([text], sp)[0]
        audio = res["audio"]
        a = (
            np.frombuffer(audio, dtype=np.float32)
            if isinstance(audio, (bytes, bytearray))
            else np.asarray(audio, dtype=np.float32).reshape(-1)
        )
        pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
        sr = int(res.get("sample_rate", 44100))
        with wave.open(f"{OUT}/cuda_clone_item{i}.wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        peak = float(np.max(np.abs(a))) if a.size else 0.0
        rms = float(np.sqrt(np.mean(a**2))) if a.size else 0.0
        lead = 0.0
        above = np.abs(a) >= 0.01
        if above.any():
            lead = int(np.argmax(above)) / sr
        print(
            f"cuda clone item{i}: frames={len(res['audio_tokens'])} "
            f"eos_frame={res['eos_frame']} dur={a.size/sr:.2f}s peak={peak:.3f} "
            f"rms={rms:.4f} lead_sil={lead*1000:.0f}ms -> cuda_clone_item{i}.wav",
            flush=True,
        )
    print("DONE cuda-clone")


if __name__ == "__main__":
    main()
