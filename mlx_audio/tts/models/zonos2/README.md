# ZONOS2

Zyphra's sparse-MoE, multi-codebook autoregressive TTS. ~8B MoE backbone (≈900M
active), 9-codebook **44.1 kHz DAC** decode, raw **UTF-8 byte** text input, and
**voice cloning** from a reference clip. Powers `model_type: "zonos2"`.

- **Original model:** [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2) · [blog](https://www.zyphra.com/our-work/zonos2)

## Highlights

- Sparse **MoE** (16 experts, top-1; top-2 at layer 26; MLP + EDA router, "sonic" `w13` experts)
- Attention with per-head QK-norm, learnable `temp`, sigmoid `gater`; **interleaved RoPE**; fused-residual RMSNorm; logit soft-cap
- Multi-codebook **delay pattern** (`shear_up` before DAC decode)
- Full sampler: temperature / top-k / top-p / min-p + windowed per-codebook **repetition penalty**
- **Voice cloning** — ECAPA-TDNN x-vector → `speaker_lda` → `speaker_projection`, injected at the speaker slot
- **Quality conditioning** — clean SNR / bandwidth / silence buckets (drops the default background noise)

## Convert weights

```bash
python -m mlx_audio.tts.models.zonos2.convert \
  --model Zyphra/ZONOS2 --output ./zonos2-mlx --dtype bfloat16
```

## Usage (Python)

```python
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.models.zonos2.zonos2 import Model

model = Model.from_local("./zonos2-mlx")

# Clean, full-quality conditioning (the default conditioning is intentionally noisy).
CLEAN = {"estimated_snr": 11, "estimated_bandlimit_hz": 7,
         "leading_silence_s": 0, "trailing_silence_s": 0, "lufs": 7, "max_pause": 0}

codes, wav = model.generate(
    "Hello, this is ZONOS two on MLX.",
    temperature=1.15, top_k=106, min_p=0.18, repetition_penalty=1.2,
    quality_buckets=CLEAN, max_frames=512,
)
audio_write("zonos2.wav", wav, 44100)
```

### Voice cloning

Pass an ECAPA speaker embedding (compute it once from a clean reference clip via
`speaker_encoder`) to clone that voice:

```python
codes, wav = model.generate("...", speaker_embedding=spk_emb, quality_buckets=CLEAN, max_frames=512)
```

## Parity (verified vs the PyTorch/CUDA reference)

- **Teacher-forced logit** argmax-agreement **98.9 %** — every residual mismatch is a
  sub-noise bf16 (MLX-Metal vs CUDA-flashinfer) reduction-order tie, not a logic bug.
- **DAC decode** byte-compare: identical codes → MLX-DAC vs torch-DAC waveform **corr 0.9998**.
- **Speaker encoder**: ECAPA embedding **cosine 1.0000** vs torch.

## Known limitation (model, not the port)

Output is band-limited to ~8–12 kHz ("old-mic" timbre). This is **identical in the
CUDA original** (the same codes decode byte-for-byte), i.e. a ceiling of the ZONOS2
model + DAC at 9 codebooks — not an MLX-port artifact.

## Notes / TODO

- `Model.generate(...)` returns `(codes, waveform)`. Conforming to the standard
  `mlx_audio.tts.load()` / `GenerationResult` streaming interface is a follow-up.
- NeMo text normalization is optional (used if `nemo_text_processing` is importable).
