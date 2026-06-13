"""STEP A (plain half): capture CLEAN-conditioned prompts + clean CUDA audio.

Run on pc.lan (CUDA, zonos2 env). Writes, for each text:
  - clean_prompt_item{i}.npy : the EXACT input token tensor _tokenize_one(text,
    quality_buckets=CLEAN_QB) produces (carries the clean quality conditioning).
  - default_prompt_item{i}.npy : the default (noisy) prompt, for diffing.
  - cuda_clean_item{i}.wav : clean plain CUDA audio (seed 42, CLEAN_QB).
Plus clean_conditioning.json with the model conditioning config so the MLX side can
reproduce the token math byte-for-byte.

The clean VOICE-CLONE capture lives in 09_capture_clean_clone.py and MUST run as a
SEPARATE process: the offline engine's TP/distributed info is a process-global
singleton, so a second TTSLLM in the same process raises "TP info has been set".
"""

import json
import os
import wave

import numpy as np
from zonos2.message import TTSSamplingParams
from zonos2.tts import TTSLLM
from zonos2.tts.llm import DEFAULT_QUALITY_BUCKETS

OUT = "/mnt/d/Projects/zonos2-parity/clean"
os.makedirs(OUT, exist_ok=True)

TEXTS = [
    "Hello world, this is a parity test.",
    "The quick brown fox jumps over the lazy dog.",
    "Numbers like 42 and dates such as June 13th should normalize cleanly.",
]
# CLEAN quality buckets (indices). lufs=7 (-23..-18.5) avoids the lufs=8 clipping.
CLEAN_QB = {
    "lufs": 7,
    "estimated_snr": 11,
    "max_pause": 0,
    "estimated_bandlimit_hz": 7,
    "leading_silence_s": 0,
    "trailing_silence_s": 0,
}


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
    # ---- plain TTS (also used for prompt capture) ----
    tts = TTSLLM(model_path="Zyphra/ZONOS2", decode_audio=True)
    from zonos2.tts.prompt import _SILENCE_TOKENS_0_2S

    cond = dict(
        n_codebooks=int(tts.n_codebooks),
        audio_pad_id=int(tts.audio_pad_id),
        text_vocab=int(tts.text_vocab),
        speaking_rate_num_buckets=int(tts.speaking_rate_num_buckets),
        quality_features=list(tts.quality_features),
        quality_bucket_counts=[int(c) for c in tts.quality_bucket_counts],
        speaker_background_num_buckets=int(tts.speaker_background_num_buckets),
        accurate_mode_num_buckets=int(tts.accurate_mode_num_buckets),
        default_quality_buckets=dict(DEFAULT_QUALITY_BUCKETS),
        clean_quality_buckets=dict(CLEAN_QB),
        resolved_clean=tts._resolve_quality_buckets(CLEAN_QB),
        resolved_default=tts._resolve_quality_buckets(None),
        silence_tokens_0_2s=_SILENCE_TOKENS_0_2S,
    )
    with open(os.path.join(OUT, "clean_conditioning.json"), "w") as f:
        json.dump(cond, f, indent=2)
    print("COND:", json.dumps(cond), flush=True)

    for i, text in enumerate(TEXTS):
        pe = (
            tts._tokenize_one(text, quality_buckets=CLEAN_QB)
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        pd = tts._tokenize_one(text).cpu().numpy().astype(np.int32)
        np.save(os.path.join(OUT, "clean_prompt_item%d.npy" % i), pe)
        np.save(os.path.join(OUT, "default_prompt_item%d.npy" % i), pd)
        head = pe[:10, -1].tolist()
        print(
            "PROMPT item%d: clean=%s default=%s clean_textcol_head=%s"
            % (i, pe.shape, pd.shape, head),
            flush=True,
        )
        r = tts.generate([text], TTSSamplingParams(seed=42), quality_buckets=CLEAN_QB)[
            0
        ]
        sr = int(r.get("sample_rate", 44100))
        save_wav(os.path.join(OUT, "cuda_clean_item%d.wav" % i), r["audio"], sr)
        s = stat(r["audio"], sr)
        nf = len(r.get("audio_tokens", []))
        ef = r.get("eos_frame")
        print(
            "CUDA-CLEAN item%d: frames=%d eos=%s dur=%.2fs peak=%.3f rms=%.4f "
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
    print(
        "DONE capture-clean (plain); run 09_capture_clean_clone.py for the clone",
        flush=True,
    )


if __name__ == "__main__":
    main()
