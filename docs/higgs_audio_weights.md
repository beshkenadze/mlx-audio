# HiggsAudioV2 Tokenizer — Weight Inspection

Source: `k2-fsa/OmniVoice` → `audio_tokenizer/model.safetensors`
Method: safetensors range-request (first 8 + 63408 bytes, no full download)
Total keys: **527**  All weights: **F32**

---

## §1 Top-level key prefixes

| Prefix | Key count | Role |
|---|---|---|
| `acoustic_decoder` | 110 | DAC-style waveform decoder (transposed-conv upsampler) |
| `acoustic_encoder` | 110 | DAC-style waveform encoder (strided-conv downsampler) |
| `decoder_semantic` | 14 | Convolutional semantic feature decoder |
| `encoder_semantic` | 13 | Convolutional semantic feature encoder |
| `fc` | 2 | 1024→1024 projection (bottleneck) |
| `fc1` | 2 | 1024→768 projection (acoustic→semantic space) |
| `fc2` | 2 | 1024→256 projection (acoustic→quantizer input) |
| `quantizer` | 64 | Residual vector quantizer (8 codebooks) |
| `semantic_model` | 210 | Frozen HuBERT-base encoder (feature extractor + transformer) |

---

## §2 Representative shapes per prefix

### acoustic_encoder

| Key | Shape | Notes |
|---|---|---|
| `acoustic_encoder.conv1.weight` | `[64, 1, 7]` | Input stem: 1-ch waveform → 64 channels |
| `acoustic_encoder.block.0.conv1.weight` | `[128, 64, 16]` | Stride-8 downsampling block |
| `acoustic_encoder.block.1.conv1.weight` | `[256, 128, 10]` | Stride-5 downsampling block |
| `acoustic_encoder.block.2.conv1.weight` | `[512, 256, 8]` | Stride-4 downsampling block |
| `acoustic_encoder.block.3.conv1.weight` | `[1024, 512, 4]` | Stride-2 downsampling block |
| `acoustic_encoder.block.4.conv1.weight` | `[2048, 1024, 6]` | Stride-3 downsampling block |
| `acoustic_encoder.conv2.weight` | `[256, 2048, 3]` | Output projection: 2048 → 256 |
| `acoustic_encoder.block.0.res_unit1.conv1.weight` | `[64, 64, 7]` | Residual unit dilated conv |
| `acoustic_encoder.block.N.snake1.alpha` | `[1, C, 1]` | Snake activation per block |

### acoustic_decoder

| Key | Shape | Notes |
|---|---|---|
| `acoustic_decoder.conv1.weight` | `[1024, 256, 7]` | Input stem: 256 → 1024 channels |
| `acoustic_decoder.block.0.conv_t1.weight` | `[1024, 512, 16]` | Stride-8 transposed-conv upsampling |
| `acoustic_decoder.block.1.conv_t1.weight` | `[512, 256, 10]` | Stride-5 upsampling |
| `acoustic_decoder.block.2.conv_t1.weight` | `[256, 128, 8]` | Stride-4 upsampling |
| `acoustic_decoder.block.3.conv_t1.weight` | `[128, 64, 4]` | Stride-2 upsampling |
| `acoustic_decoder.block.4.conv_t1.weight` | `[64, 32, 6]` | Stride-3 upsampling |
| `acoustic_decoder.conv2.weight` | `[1, 32, 7]` | Output stem: 32 → 1-ch waveform |

### quantizer

| Key | Shape | Notes |
|---|---|---|
| `quantizer.quantizers.{0..7}.codebook.embed` | `[1024, 64]` | 8 codebooks × 1024 entries × 64-dim |
| `quantizer.quantizers.{0..7}.project_in.weight` | `[64, 1024]` | Project 1024-d → 64-d codebook space |
| `quantizer.quantizers.{0..7}.project_out.weight` | `[1024, 64]` | Project 64-d → 1024-d residual space |

### encoder_semantic / decoder_semantic

| Key | Shape | Notes |
|---|---|---|
| `encoder_semantic.conv.weight` | `[768, 768, 3]` | Input conv |
| `encoder_semantic.conv_blocks.{0,1}.conv.weight` | `[768, 768, 3]` | 2 conv blocks |
| `decoder_semantic.conv1.weight` | `[768, 768, 3]` | Mirror decoder input |
| `decoder_semantic.conv_blocks.{0,1}.conv.weight` | `[768, 768, 3]` | 2 conv blocks |

### fc projections

| Key | Shape | Notes |
|---|---|---|
| `fc.weight` | `[1024, 1024]` | Bottleneck self-projection |
| `fc1.weight` | `[768, 1024]` | Acoustic (1024) → semantic (768) |
| `fc2.weight` | `[256, 1024]` | Acoustic (1024) → quantizer input (256) |

### semantic_model (HuBERT-base)

| Key | Shape | Notes |
|---|---|---|
| `semantic_model.feature_extractor.conv_layers.0.conv.weight` | `[512, 1, 10]` | Layer-norm CNN, stride-5 |
| `semantic_model.feature_extractor.conv_layers.{1..6}.conv.weight` | `[512, 512, K]` | K ∈ {3,3,3,3,2,2}, group-norm CNN |
| `semantic_model.feature_projection.projection.weight` | `[768, 512]` | CNN-out 512 → transformer 768 |
| `semantic_model.encoder.layers.{0..11}.attention.q_proj.weight` | `[768, 768]` | 12-layer transformer |
| `semantic_model.encoder.layers.{0..11}.feed_forward.intermediate_dense.weight` | `[3072, 768]` | FFN width 3072 |
| `semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1` | `[768, 48, 128]` | Relative position conv |
| `semantic_model.encoder.layer_norm.weight` | `[768]` | Final layer norm |

---

## §3 Confirmed encoder topology (channels and strides)

### Acoustic encoder (DAC-style)

```
Input: waveform [B, 1, T]  @ 24 kHz
  conv1 [64, 1, 7]        → [B, 64, T]          (no downsampling, causal pad)
  block.0 [128, 64, 16]   stride=8  → [B, 128, T/8]
    res_unit×3: [64, 64, 7]
  block.1 [256, 128, 10]  stride=5  → [B, 256, T/40]
    res_unit×3: [128, 128, 7]
  block.2 [512, 256, 8]   stride=4  → [B, 512, T/160]
    res_unit×3: [256, 256, 7]
  block.3 [1024, 512, 4]  stride=2  → [B, 1024, T/320]
    res_unit×3: [512, 512, 7]
  block.4 [2048, 1024, 6] stride=3  → [B, 2048, T/960]
    res_unit×3: [1024, 1024, 7]
  snake [1, 2048, 1]
  conv2 [256, 2048, 3]    → [B, 256, T/960]     (latent)
```

**Total hop length (acoustic): 8 × 5 × 4 × 2 × 3 = 960 samples**
At 24 kHz this gives ~40 tokens/second.

Channel sequence: `1 → 64 → 128 → 256 → 512 → 1024 → 2048 → 256`
(encoder_hidden_size=64, decoder_hidden_size=1024 per config, hidden_size=256 is the latent dim)

Strides: `[8, 5, 4, 2, 3]` (matches `downsampling_ratios` in config.json)

### Acoustic decoder (mirror)

```
  conv1 [1024, 256, 7]              (256 latent → 1024)
  block.0 conv_t1 [1024, 512, 16]  stride=8  upsample
  block.1 conv_t1 [512, 256, 10]   stride=5  upsample
  block.2 conv_t1 [256, 128, 8]    stride=4  upsample
  block.3 conv_t1 [128, 64, 4]     stride=2  upsample
  block.4 conv_t1 [64, 32, 6]      stride=3  upsample
  snake [1, 32, 1]
  conv2 [1, 32, 7]                  → waveform
```

### Quantizer (RVQ)

- 8 active codebooks (config has `n_codebooks=9`, quantizer weights cover indices 0–7, index 8 is likely reserved for training dropout)
- Codebook size: 1024 entries
- Codebook dimension: 64 (projected from/to 1024-d latent via `project_in` / `project_out`)

### Semantic encoder (HuBERT-base-like)

```
feature_extractor: 7-layer CNN
  layer 0: [512, 1, 10] stride=5  (layer-norm, gelu)
  layer 1: [512, 512, 3] stride=2 (group-norm)
  layers 2–4: [512, 512, 3] stride=2
  layers 5–6: [512, 512, 2] stride=2
Total stride = 5×2^6 = 320  → 16 kHz input / 320 = 50 tokens/sec

feature_projection: [768, 512]
encoder: 12 transformer layers
  hidden_size=768, intermediate=3072, heads=12
  positional: relative pos conv (num_conv_pos_embeddings=128, groups=16)
```

**Semantic hop length: 320 samples @ 16 kHz = 20 ms per frame**

---

## §4 HuBERT decision

**Branch B — HuBERT IS REQUIRED.**

The `semantic_model.*` subtree (210 keys, ~86 MB) is a fully-embedded HuBERT-base
transformer with all weights stored in the tokenizer checkpoint. It is **not** an
external dependency: the entire model (CNN feature extractor + 12-layer transformer
+ feature projection) lives inside `model.safetensors`.

The architecture exactly matches HuBERT-base:
- 7-layer CNN feature extractor (stride product = 320)
- 12 transformer layers, hidden=768, FFN=3072, heads=12
- Same layer naming as `facebook/hubert-base-ls960`

The `semantic_model` is used to extract discrete semantic tokens from the waveform
prior to quantization, as confirmed by the `encoder_semantic`/`decoder_semantic`
adapter convolutions (768-dim) and the `fc1` bridge (1024→768).

To port this to MLX you must implement (or load) a HuBERT-base encoder. The weights
are already present in the checkpoint — no separate download is needed, but the
class must exist to load them.

---

## §5 `sanitize()` key mapping table

This table maps the PyTorch safetensors key names to the expected MLX module path.
The MLX port should use the same key names verbatim (no renaming needed) since the
checkpoint is already a standard HuggingFace model saved with `save_pretrained`.

| Checkpoint key pattern | MLX module path | Notes |
|---|---|---|
| `acoustic_encoder.conv1.{weight,bias}` | `acoustic_encoder.conv1.{weight,bias}` | Identity |
| `acoustic_encoder.block.N.conv1.{weight,bias}` | `acoustic_encoder.block.N.conv1.{weight,bias}` | Identity |
| `acoustic_encoder.block.N.res_unitM.conv{1,2}.{weight,bias}` | same | Identity |
| `acoustic_encoder.block.N.snake1.alpha` | same | Snake activation param |
| `acoustic_encoder.conv2.{weight,bias}` | `acoustic_encoder.conv2.{weight,bias}` | Identity |
| `acoustic_decoder.*` | `acoustic_decoder.*` | Mirror of encoder, identity |
| `quantizer.quantizers.N.codebook.embed` | `quantizer.quantizers.N.codebook.embed` | Identity |
| `quantizer.quantizers.N.codebook.embed_avg` | **drop** | EMA training buffer, unused at inference |
| `quantizer.quantizers.N.codebook.cluster_size` | **drop** | EMA training buffer, unused at inference |
| `quantizer.quantizers.N.codebook.inited` | **drop** | Training flag, unused at inference |
| `quantizer.quantizers.N.project_in.{weight,bias}` | same | Identity |
| `quantizer.quantizers.N.project_out.{weight,bias}` | same | Identity |
| `encoder_semantic.conv.weight` | `encoder_semantic.conv.weight` | Identity (no bias) |
| `encoder_semantic.conv_blocks.N.conv.{weight,bias}` | same | Identity |
| `encoder_semantic.conv_blocks.N.res_units.M.conv{1,2}.weight` | same | Identity (no bias) |
| `decoder_semantic.*` | `decoder_semantic.*` | Mirror, identity |
| `fc.{weight,bias}` | `fc.{weight,bias}` | Identity |
| `fc1.{weight,bias}` | `fc1.{weight,bias}` | Identity |
| `fc2.{weight,bias}` | `fc2.{weight,bias}` | Identity |
| `semantic_model.feature_extractor.conv_layers.N.conv.weight` | same | Identity (no bias; conv_bias=false) |
| `semantic_model.feature_extractor.conv_layers.0.layer_norm.{weight,bias}` | same | Layer-norm on first CNN layer |
| `semantic_model.feature_projection.layer_norm.{weight,bias}` | same | Identity |
| `semantic_model.feature_projection.projection.{weight,bias}` | same | Identity |
| `semantic_model.encoder.layers.N.attention.{q,k,v}_proj.{weight,bias}` | same | Identity |
| `semantic_model.encoder.layers.N.attention.out_proj.{weight,bias}` | same | Identity |
| `semantic_model.encoder.layers.N.feed_forward.intermediate_dense.{weight,bias}` | same | Identity |
| `semantic_model.encoder.layers.N.feed_forward.output_dense.{weight,bias}` | same | Identity |
| `semantic_model.encoder.layers.N.layer_norm.{weight,bias}` | same | Pre-attention LN |
| `semantic_model.encoder.layers.N.final_layer_norm.{weight,bias}` | same | Post-FFN LN |
| `semantic_model.encoder.layer_norm.{weight,bias}` | same | Top-level encoder LN |
| `semantic_model.encoder.pos_conv_embed.conv.bias` | same | Positional conv bias |
| `semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0` | **needs reshape/merge** | Weight norm: norm vector `[1,1,128]` |
| `semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1` | **needs reshape/merge** | Weight norm: direction `[768,48,128]` → compute `weight = original1 * (original0 / ||original1||)` to get `[768, 768, 128]` (groups=16) |

### Weight norm reconstruction for pos_conv_embed

PyTorch `parametrize` stores `weight_g` (original0) and `weight_v` (original1).
The effective weight is:

```python
# original0: [1, 1, 128]   (g, the norm scale)
# original1: [768, 48, 128] (v, the direction; 48 = 768/groups=16)
# Reconstruct:
v_norm = mx.sqrt(mx.sum(original1 ** 2, axis=(1, 2), keepdims=True))  # [768, 1, 1]
weight = original1 / v_norm * original0  # [768, 48, 128]
# This is a grouped conv weight: out=768, in_per_group=48, kernel=128, groups=16
```
