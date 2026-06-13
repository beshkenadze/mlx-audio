"""STEP A (clone half): clean CUDA voice-clone audio (run as its OWN process).

The TP/distributed info is a process-global singleton in the offline engine, so a
second TTSLLM in the same process raises "TP info has been set". This script
therefore generates ONLY the clean voice-clone wavs (AmericanMale ECAPA embedding +
CLEAN_QB) in a fresh process. The plain clean capture lives in 08_capture_clean.py.
"""

import os
import wave

import numpy as np
import torch
from zonos2.message import TTSSamplingParams, TTSUserMsg
from zonos2.tts import TTSLLM

OUT = "/mnt/d/Projects/zonos2-parity/clean"
EMB_NPY = "/mnt/d/Projects/zonos2-parity/spk_emb_americanmale.npy"
os.makedirs(OUT, exist_ok=True)

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


class ClonedTTSLLM(TTSLLM):
    def __init__(self, *a, speaker_embedding=None, **k):
        super().__init__(*a, **k)
        self._clone_emb = speaker_embedding

    def offline_receive_msg(self, blocking=False):
        msgs = super().offline_receive_msg(blocking=blocking)
        for m in msgs:
            if isinstance(m, TTSUserMsg) and self._clone_emb is not None:
                m.speaker_embedding = self._clone_emb
                m.speaker_token_position = 0
                m.clean_speaker_background = True
                m.accurate_mode = False
        return msgs


def to_f32(audio):
    if isinstance(audio, (bytes, bytearray)):
        return np.frombuffer(audio, dtype=np.float32).reshape(-1)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def stat(audio, sr):
    a = to_f32(audio)
    win = int(0.02 * sr)
    floor = 0.0
    if len(a) >= win:
        e = np.array(
            [np.sqrt(np.mean(a[i : i + win] ** 2)) for i in range(0, len(a) - win, win)]
        )
        floor = float(np.percentile(e, 10))
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    rms = float(np.sqrt(np.mean(a**2))) if a.size else 0.0
    above = np.abs(a) >= 0.01
    lead = trail = (a.size / sr) if a.size else 0.0
    if above.any():
        first = int(np.argmax(above))
        last = int(len(above) - 1 - np.argmax(above[::-1]))
        lead = first / sr
        trail = (len(a) - 1 - last) / sr
    return dict(
        dur=a.size / sr, peak=peak, rms=rms, floor=floor, lead=lead, trail=trail
    )


def save_wav(path, audio, sr):
    a = to_f32(audio)
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def main():
    emb = torch.from_numpy(np.load(EMB_NPY).astype(np.float32))
    print(
        "clone embedding shape=%s norm=%.3f" % (tuple(emb.shape), emb.norm().item()),
        flush=True,
    )
    ctts = ClonedTTSLLM(
        model_path="Zyphra/ZONOS2", decode_audio=True, speaker_embedding=emb
    )
    for i, text in enumerate(TEXTS):
        r = ctts.generate([text], TTSSamplingParams(seed=42), quality_buckets=CLEAN_QB)[
            0
        ]
        sr = int(r.get("sample_rate", 44100))
        save_wav(os.path.join(OUT, "cuda_clean_clone_item%d.wav" % i), r["audio"], sr)
        s = stat(r["audio"], sr)
        nf = len(r.get("audio_tokens", []))
        ef = r.get("eos_frame")
        print(
            "CUDA-CLEAN-CLONE item%d: frames=%d eos=%s dur=%.2fs peak=%.3f rms=%.4f "
            "floor=%.5f lead=%.0fms trail=%.0fms"
            % (
                i,
                nf,
                ef,
                s["dur"],
                s["peak"],
                s["rms"],
                s["floor"],
                s["lead"] * 1000,
                s["trail"] * 1000,
            ),
            flush=True,
        )
    print("DONE capture-clean-clone", flush=True)


if __name__ == "__main__":
    main()
