# ZONOS2 MLX Port — Shared Contract (read before coding)

This file is the single source of truth for every ZONOS2 sub-module so the
independently-built pieces integrate cleanly in Phase D. Plan:
`/Users/akira/.claude/plans/witty-imagining-wren.md`.

## Ground rules
- **Ground truth = the upstream PyTorch source + the checkpoint tensor shapes**, not
  blog posts or this summary. When a shape/logic detail is ambiguous, fetch the named
  reference file and mirror it exactly. (Lesson from prior ports: checkpoint metadata
  can lie; tensor names/shapes are authoritative.)
- **Do NOT edit** `__init__.py`, `config.py`, or the (future) `zonos2.py` — the
  coordinator owns integration. Create only your own `<unit>.py` + `tests/test_<unit>.py`.
- Match surrounding mlx-audio style. RMSNorm via `mx.fast.rms_norm`. No `any`-typing.
- Weights load as **bf16**; norms/router/special tensors stay **fp32** (parity).
- All shapes below are `[batch, ...]`; sequence axis is `T`.

## Fetching upstream reference files (curl/WebFetch are sandbox-blocked here)
Use the JS sandbox: `ctx_execute(language:"javascript", code: ...)` with
`await fetch("https://raw.githubusercontent.com/Zyphra/ZONOS2/main/<path>")`.
Or `gh api repos/Zyphra/ZONOS2/contents/<path> --jq .content | base64 -d`.

## Config (`ZONOS2Config`, already implemented in `config.py`)
Import it: `from mlx_audio.tts.models.zonos2.config import ZONOS2Config`.
Released-checkpoint values + derived properties (use these, don't re-derive):

| field | value | | derived | value |
|---|---|---|---|---|
| n_layers | 28 | | num_qo_heads | 16 (`dim//head_dim`) |
| dim (hidden) | 2048 | | **intermediate_size** | **3072** (`round(1.5·2048)→×256`) |
| head_dim | 128 | | moe_intermediate_size | 3072 |
| n_kv_heads | 4 (GQA) | | audio_vocab/codebook | 1026 (`codebook_size+2`) |
| norm_eps | 1e-5 | | **vocab_size** | **9754** (`9·1026 + 519+1`) |
| rope_theta | 10000 | | is_moe_layer(i) | `3 ≤ i < 27` → 3..26 MoE; 0/1/2/27 dense |
| max_seqlen | 6144 | | num_experts_per_tok(i) | 1, except **i==26 → 2** |
| n_codebooks | 9 | | loss_softcap | 15.0 |
| codebook_size | 1024 | | eoa_id / audio_pad_id | 1024 / 1025 |
| text_vocab | 519 | | | |
| moe_n_experts | 16 | | moe_router_dim | 128 |

## Tensor conventions
- Audio codes input: `[B, n_codebooks=9, T]` (int). Text tokens: `[B, T]` (UTF-8 bytes).
- Hidden state: `[B, T, dim=2048]`.
- Final logits: `[B, T, n_codebooks=9, audio_vocab=1026]` (after softcap).
- Speaker: ref-wav → 2048-D embedding → LDA → 1024-D conditioning vector.

## Checkpoint key naming (from upstream docstrings; `model.pth` is `Zyphra/ZONOS2`)
- Embedding: `multi_embedder.embedders.{0..8}.weight` (audio, num_emb=1026, pad_idx=1025),
  then a text embedder (pad_idx=`text_vocab`). Summed per position.
- Output head: `multi_output.weight` `[vocab_out, hidden]`, `F.linear` (no bias).
- Attention: `layers.{N}.attention.{wq,wkv,wo,...}` (GQA, head_dim 128, kv=4).
- Dense FFN: `layers.{N}.feed_forward.w_in.weight` **rank-3 `[2, inter, hidden]`** (fused
  gate+up; `[0]`=up "h", `[1]`=gate w/ SiLU), `feed_forward.w_out.weight` `[hidden, inter]`.
- MoE experts ("sonic"): fused `w13` rank-3 `[experts, 2·inter, hidden]`, **interleaved**:
  `gate = w13[:, 0::2, :]`, `up = w13[:, 1::2, :]`; `w2`→`down_proj`.
- MoE router: `router.down_proj.{weight,bias}` (hidden→128), `router.rmsnorm_eda.weight`,
  `router.router_mlp.{0,2,4}.weight` + `{0,2}.bias` (GELU MLP → 16 logits),
  `router.balancing_biases` `[16]`, `router.router_states_scale` (EDA layers only;
  `use_eda=True` for every MoE layer except the first MoE layer = layer 3).
- Norms: `layers.{N}.attention_norm.weight`, `layers.{N}.ffn_norm.weight`, `norm.weight`.
- Weight-norm parametrization: strip `*.parametrizations.X.original*` → `*.X*` if present.

## Per-unit API each module MUST expose (so Phase D wiring is mechanical)
> All take a `ZONOS2Config` named `config` where relevant. Keep classes `nn.Module`.

1. **`layers.py`**
   - `class RMSNorm(nn.Module)` `(dims, eps)` → `mx.fast.rms_norm`.
   - `class RotaryEmbedding` for head_dim=128, base=`rope_theta`; helper `apply_rotary`.
   - `class Attention(nn.Module)(config, layer_id)` `__call__(x, *, mask=None, cache=None, offset=0) -> x`
     (GQA q=16/kv=4 heads, head_dim 128, no bias unless ckpt has it).
   - `class FeedForward(nn.Module)(config)` dense fused-`w_in` SwiGLU → `[B,T,dim]`.
   - `def softcap(x, cap) -> x` = `cap * tanh(x / cap)`.
2. **`moe.py`**
   - `class Router(nn.Module)(config, layer_id)` `__call__(x) -> (indices[B,T,k], weights[B,T,k])`
     (down_proj → rmsnorm_eda → GELU MLP → topk via `mx.argpartition`; honor EDA + balancing_biases).
   - `class MoEFeedForward(nn.Module)(config, layer_id)` `__call__(x) -> [B,T,dim]`; experts via
     `mlx_lm.models.switch_layers.SwitchGLU` (`mx.gather_mm`).
   - `def split_sonic_w13(w13) -> (gate, up)` (gate=even rows, up=odd rows) — used by `convert.py`.
3. **`tokenizer.py`**
   - `class ZONOS2Tokenizer` with `encode(text:str) -> list[int]` (UTF-8 bytes + specials,
     vocab 519) and `decode`. Optional `normalize(text)` hook: use `nemo_text_processing`
     if importable else identity (passthrough). No hard pynini dependency.
4. **`speaker.py`**
   - `class SpeakerLDA(nn.Module)(config)` `__call__(emb_2048) -> [.,1024]`.
   - `def speaking_rate_bucket(value, config) -> int`, `def quality_bucket(feature, value, config) -> int`
     (parse the `"lo-hi"` / `"v+"` bucket strings from config; mirror upstream boundaries).
5. **`speaker_encoder.py`** (Qwen3-Voice-Embedding-1.7B → 2048-D)
   - `class VoiceMelFrontend(nn.Module)` (n_fft 1024, hop 256, win 1024, 128 mels, f_min 0,
     f_max 12000, slaney norm+scale, **center=False**, reflect-pad `(n_fft-hop)//2`, `log(clamp(mel,1e-5))`,
     transpose to `[B, frames, 128]`).
   - `class Qwen3VoiceEmbedding(nn.Module)` reusing the qwen3 backbone
     (`mlx_audio/tts/models/qwen3/qwen3.py` or `mlx_lm`), `__call__(wav, sample_rate) -> [B, frames, 2048]`.
     HF: `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`, target SR 24000.
6. **`convert.py`**
   - `def convert(pth_path, out_dir, dtype="bfloat16") -> Path` — torch `model.pth` → MLX
     `model.safetensors`: strip weight-norm parametrization, split fused dense `w_in` and
     sonic `w13` (use `split_sonic_w13`), map GQA, dtype policy above.
   - `def convert_voice_embedder(hf_repo, out_dir, dtype) -> Path` for the Qwen3 embedder.
   - Test on a **synthetic state-dict fixture** (real 8B weights live on pc.lan; Phase D validates).
7. **`sampling.py`**
   - `def make_codebook_sampler(temperature, top_k, top_p, softcap) -> callable(logits)->tokens`,
     per-codebook. Reuse `mlx_lm.sample_utils` where possible. Greedy when temperature==0.

## Reference-file map (upstream `python/zonos2/...`)
| unit | mirror |
|---|---|
| layers | `layers/attention.py`, `layers/rotary.py`, `layers/norm.py`, `models/zonos2.py` (`Attention`,`FeedForward`) |
| moe | `models/zonos2.py` (`Router`,`RouterMLP`,`FusedGroupedExperts`,`MoEFeedForward`,`_convert_sonic_w13_to_gate_up`) |
| tokenizer | `message/tokenizer.py`, `tokenizer/textnorm.py` |
| speaker | `models/speaker_lda.py`, `models/config.py` (bucket strings) |
| speaker_encoder | `models/speaker_cloning.py` (`Qwen3SpeakerEmbedding`) |
| convert | `models/zonos2.py` load_state_dict, `models/weight.py` |
| sampling | `tts/sampler.py` |

## Verify
`cd /Volumes/DATA/mlx-audio && .venv/bin/python -m pytest mlx_audio/tts/models/zonos2/tests/test_<unit>.py -q`
(synthetic shapes; no GPU, no CUDA fixtures — full CUDA parity is the coordinator Phase-D gate).
