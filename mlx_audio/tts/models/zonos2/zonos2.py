# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
"""ZONOS2 top-level model (MLX port).

Wires the merged sub-modules (``layers``, ``moe``, ``tokenizer``, ``sampling``,
``speaker``) into a text -> DAC-tokens -> audio pipeline that mirrors the
upstream PyTorch forward (``python/zonos2/models/zonos2.py``) and the offline
TTS decode loop (``python/zonos2/tts/{llm,sequence,sampler}.py`` and
``python/zonos2/scheduler``).

The backbone is the upstream ``Zonos2ForCausalLM``:

* ``multi_embedder`` -- one embedding table per frame column; the per-position
  embedding is the SUM over all ``frame_width`` columns (9 audio codebooks + 1
  text column). No padding mask.
* ``emb_norm`` -- an RMSNorm (no learnable weight upstream:
  ``elementwise_affine=False``) applied to the summed embedding before the
  layer stack, with ``residual=None`` (the residual it returns is discarded).
* 28 ``TransformerBlock``s with FUSED add-norm residual threading
  (``RMSNormFused``): ``x, residual = attention_norm(x, residual)`` /
  ``x, residual = ffn_norm(x, residual)``. Layers 3..26 are MoE and thread
  ``router_states`` for the EDA blend; the remaining layers are dense.
* ``out_norm`` -- a final fused add-norm that combines the trailing residual.
* ``multi_output`` -- a single ``[n_codebooks * audio_vocab, hidden]`` head;
  reshaped to ``[.., n_codebooks, audio_vocab]`` then soft-capped
  (``loss_softcap``).

Generation is autoregressive with a KV cache. Each step samples 9 audio codes;
the next input row is ``[cb0, .., cb8, text_pad]`` (the text column is the text
padding id ``text_vocab``). The codebooks carry a learned delay pattern; the raw
sampled frames are returned as-is and only de-sheared (``vocoder.shear_up``)
before DAC decode, matching the upstream offline ``generate``.

Reference: https://github.com/Zyphra/ZONOS2 -> python/zonos2/models/zonos2.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import KVCache

from mlx_audio.tts.models.zonos2.config import ZONOS2Config
from mlx_audio.tts.models.zonos2.layers import Attention, FeedForward, softcap
from mlx_audio.tts.models.zonos2.moe import MoEFeedForward
from mlx_audio.tts.models.zonos2.sampling import make_codebook_sampler
from mlx_audio.tts.models.zonos2.tokenizer import ZONOS2Tokenizer


class RMSNormFused(nn.Module):
    """Fused add-norm RMSNorm mirroring upstream ``RMSNormFused``.

    ``__call__(x, residual)`` semantics (matching flashinfer
    ``fused_add_rmsnorm``):

    * ``residual is None`` (first block): return ``(rmsnorm(x), x)`` -- the raw
      input becomes the running residual.
    * otherwise: ``residual = residual + x``; return
      ``(rmsnorm(residual), residual)``.

    **Precision (parity-critical).** The residual stream and the RMSNorm math run
    in fp32. CUDA flashinfer fuses add+rmsnorm and accumulates the residual in a
    higher-precision buffer; if the residual is kept in bf16, the output logits
    land on exact bf16 ties (e.g. two codebook tokens with identical rounded
    logits) and greedy argmax becomes non-deterministic, diverging from the
    reference. Keeping the residual in fp32 reproduces the CUDA argmax exactly.
    The *normalized* output is cast back to ``compute_dtype`` (bf16) so the
    downstream attention/FFN matmuls run in bf16 like upstream.

    When ``elementwise_affine`` is False there is no learnable weight (a unit
    weight is used); this matches the upstream ``emb_norm`` which is not present
    in the checkpoint.
    """

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        compute_dtype: Optional[mx.Dtype] = None,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # ``None`` -> cast the normed output back to the incoming activation
        # dtype (bf16 for the real model, fp32 for fp32 tests); the residual
        # stream is ALWAYS fp32. ``out_norm`` overrides this to force fp32.
        self.compute_dtype = compute_dtype
        self._dims = dims
        if elementwise_affine:
            self.weight = mx.ones((dims,))
        else:
            # Constant unit weight; not a parameter so it is excluded from the
            # checkpoint key set (upstream emb_norm has no learnable weight).
            self._ones = mx.ones((dims,))

    def _weight(self) -> mx.array:
        return self.weight if self.elementwise_affine else self._ones

    def __call__(
        self, x: mx.array, residual: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        out_dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        weight = self._weight().astype(mx.float32)
        x = x.astype(mx.float32)
        residual = x if residual is None else residual + x
        normed = mx.fast.rms_norm(residual, weight, self.eps)
        return normed.astype(out_dtype), residual


class MultiEmbedding(nn.Module):
    """Per-column embedding tables; output = SUM over all frame columns.

    Checkpoint keys: ``multi_embedder.embedders.{0..n_codebooks}.weight``.
    Embedder ``i`` embeds column ``i`` of the ``[.., frame_width]`` input.
    Columns ``0..n_codebooks-1`` are audio (vocab ``codebook_size + 2``); the
    final column is text (vocab ``text_vocab + 1``).
    """

    def __init__(self, config: ZONOS2Config):
        super().__init__()
        audio_num_emb = config.audio_vocab  # codebook_size + 2
        embedders: List[nn.Embedding] = [
            nn.Embedding(audio_num_emb, config.dim) for _ in range(config.n_codebooks)
        ]
        if config.text_vocab is not None:
            embedders.append(nn.Embedding(config.text_vocab + 1, config.dim))
        self.embedders = embedders

    def __call__(self, input_ids: mx.array) -> mx.array:
        # input_ids: [.., frame_width] integer ids.
        n_cols = input_ids.shape[-1]
        result = self.embedders[0](input_ids[..., 0])
        for i in range(1, n_cols):
            result = result + self.embedders[i](input_ids[..., i])
        return result


class MultiOutputHead(nn.Module):
    """Single linear output head (``multi_output.weight``, no bias).

    The projection runs in fp32 (weight upcast) so the per-codebook argmax is
    taken on full-precision logits; the bf16-rounded logits otherwise land on
    exact ties that make greedy decoding diverge from the CUDA reference.
    """

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.weight = mx.zeros((output_size, hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) @ self.weight.astype(mx.float32).T


class TransformerBlock(nn.Module):
    """Decoder block with fused add-norm residual threading.

    Mirrors upstream ``TransformerBlock.forward(x, residual, router_states)``:
    attention_norm -> attention -> ffn_norm -> feed_forward. MoE layers thread
    ``router_states`` for the EDA blend; dense layers reset it to ``None``.
    """

    def __init__(self, config: ZONOS2Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(config, layer_id)
        self.attention_norm = RMSNormFused(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNormFused(config.dim, eps=config.norm_eps)

        self.is_moe = config.is_moe_layer(layer_id)
        if self.is_moe:
            self.feed_forward = MoEFeedForward(config, layer_id)
        else:
            self.feed_forward = FeedForward(config)

    def __call__(
        self,
        x: mx.array,
        residual: Optional[mx.array] = None,
        router_states: Optional[mx.array] = None,
        *,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        x, residual = self.attention_norm(x, residual)
        x = self.attention(x, mask=mask, cache=cache)
        x, residual = self.ffn_norm(x, residual)
        if self.is_moe:
            x, router_states = self.feed_forward(
                x, router_states, return_router_states=True
            )
        else:
            x = self.feed_forward(x)
            router_states = None
        return x, residual, router_states


class Zonos2Backbone(nn.Module):
    """ZONOS2 causal LM backbone (upstream ``Zonos2ForCausalLM``).

    Checkpoint naming (no ``model.`` prefix):
      ``multi_embedder.*``, ``layers.{N}.*``, ``out_norm.weight``,
      ``multi_output.weight``. ``emb_norm`` has no learnable weight.
    """

    def __init__(self, config: ZONOS2Config):
        super().__init__()
        self.config = config
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.audio_vocab

        self.multi_embedder = MultiEmbedding(config)
        self.emb_norm = RMSNormFused(
            config.dim, eps=config.norm_eps, elementwise_affine=False
        )
        self.layers = [
            TransformerBlock(config, layer_id) for layer_id in range(config.n_layers)
        ]
        # out_norm emits fp32 so the output head sees full-precision hidden
        # states (parity-critical for argmax ties; see RMSNormFused).
        self.out_norm = RMSNormFused(
            config.dim, eps=config.norm_eps, compute_dtype=mx.float32
        )
        self.multi_output = MultiOutputHead(
            config.dim, self.audio_vocab * self.n_codebooks
        )

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Optional[List[Optional[KVCache]]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Return hidden states ``[B, T, hidden]`` for ``input_ids [B, T, W]``."""
        if cache is None:
            cache = [None] * len(self.layers)

        x = self.multi_embedder(input_ids)
        # emb_norm: residual=None -> rmsnorm(x); the returned residual is dropped.
        x, _ = self.emb_norm(x, None)

        if mask is None and x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(
                x.dtype
            )

        residual: Optional[mx.array] = None
        router_states: Optional[mx.array] = None
        for layer, layer_cache in zip(self.layers, cache):
            x, residual, router_states = layer(
                x, residual, router_states, mask=mask, cache=layer_cache
            )

        hidden, _ = self.out_norm(x, residual)
        return hidden

    def compute_logits(self, hidden: mx.array) -> mx.array:
        """Project hidden -> ``[.., n_codebooks, audio_vocab]`` + soft-cap."""
        logits = self.multi_output(hidden)
        *batch_dims, _ = logits.shape
        logits = logits.reshape(*batch_dims, self.n_codebooks, self.audio_vocab)
        if self.config.loss_softcap > 0:
            logits = softcap(logits, self.config.loss_softcap)
        return logits

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in self.layers]


class Model(nn.Module):
    """ZONOS2 TTS model: text -> audio codes -> waveform.

    ``config`` is the raw ``params.json`` dict (filtered by
    :meth:`ZONOS2Config.from_dict`). The DAC vocoder is loaded lazily so the
    backbone can be exercised (and unit-tested) without network access.
    """

    DAC_REPO = "mlx-community/descript-audio-codec-44khz"

    def __init__(self, config: Union[dict, ZONOS2Config]):
        super().__init__()
        if isinstance(config, ZONOS2Config):
            self.config = config
        else:
            self.config = ZONOS2Config.from_dict(config)

        self.backbone = Zonos2Backbone(self.config)
        self.tokenizer = ZONOS2Tokenizer()

        self.n_codebooks = self.config.n_codebooks
        self.audio_pad_id = self.config.audio_pad_id
        self.eoa_id = self.config.eoa_id
        self.text_vocab = self.config.text_vocab
        # Text column padding id for generated frames (mirrors sample_tts, which
        # appends `text_vocab` after the sampled audio codes).
        self.text_pad_id = self.text_vocab if self.text_vocab is not None else 0
        self.frame_width = self.n_codebooks + 1

        self._dac = None

    # ── weights ───────────────────────────────────────────────────────────────

    @property
    def dac(self):
        if self._dac is None:
            from mlx_audio.codec.models import DAC

            self._dac = DAC.from_pretrained(self.DAC_REPO)
        return self._dac

    def sanitize(self, weights: dict) -> dict:
        """Reconcile ``convert.py`` checkpoint keys with this port's modules.

        Two purely-naming adaptations bridge the converted checkpoint (which
        keeps the upstream MoE expert/router-MLP names) and the merged
        ``moe.py`` modules (which name the SwitchGLU ``experts`` and wrap the
        router MLP in a ``layers`` list):

        * ``feed_forward.switch_mlp.``  -> ``feed_forward.experts.``
          (``convert.py`` writes the de-interleaved stacked experts under
          ``switch_mlp``; :class:`MoEFeedForward` exposes them as ``experts``).
        * ``...router.router_mlp.{N}.`` -> ``...router.router_mlp.layers.{N}.``
          (upstream ``nn.Sequential`` numeric keys vs. :class:`RouterMLP`'s
          ``self.layers`` list).
        """
        out = {}
        for key, value in weights.items():
            key = key.replace("feed_forward.switch_mlp.", "feed_forward.experts.")
            key = re.sub(
                r"(\.router\.router_mlp)\.(\d+)\.",
                r"\1.layers.\2.",
                key,
            )
            out[key] = value
        return out

    def load_weights(self, weights, strict: bool = True):
        """Load weights into the backbone.

        ``weights`` may be a path, a list of ``(key, array)`` pairs, or a dict.
        Keys are addressed under ``backbone.`` so the checkpoint's bare
        ``multi_embedder.* / layers.* / out_norm.* / multi_output.*`` names map
        onto :class:`Zonos2Backbone`.
        """
        if isinstance(weights, (str, Path)):
            weights = dict(mx.load(str(weights)).items())
        elif isinstance(weights, list):
            weights = dict(weights)

        weights = self.sanitize(weights)
        prefixed = [(f"backbone.{k}", v) for k, v in weights.items()]
        super().load_weights(prefixed, strict=strict)
        return self

    @classmethod
    def from_local(cls, model_dir: Union[str, Path]) -> "Model":
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        model = cls(config)
        model.load_weights(str(model_dir / "model.safetensors"))
        model.eval()
        return model

    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "Model":
        path = Path(model_path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    str(model_path),
                    allow_patterns=["*.json", "*.safetensors"],
                )
            )
        return cls.from_local(path)

    # ── generation ──────────────────────────────────────────────────────────────

    def _next_input_row(self, audio_codes: mx.array) -> mx.array:
        """Build the next ``[1, 1, frame_width]`` input row from sampled codes.

        ``audio_codes`` is ``[n_codebooks]``; the text column is the text pad id.
        """
        text_col = mx.array([self.text_pad_id], dtype=audio_codes.dtype)
        row = mx.concatenate([audio_codes, text_col], axis=0)
        return row.reshape(1, 1, self.frame_width)

    def generate_codes(
        self,
        input_ids: mx.array,
        *,
        max_frames: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_on_eoa: bool = True,
        return_eos_frame: bool = False,
    ):
        """Autoregressive audio-code generation from a prebuilt prompt.

        Args:
            input_ids: prompt of shape ``[seq, frame_width]`` (or
                ``[1, seq, frame_width]``) of int ids.
            max_frames: number of audio frames to generate.
            temperature/top_k/top_p: per-codebook sampling controls (greedy when
                ``temperature <= 0`` or ``top_k == 1``).
            stop_on_eoa: stop early when ``eoa_id`` appears in any codebook of a
                sampled frame (delay-aware countdown of ``n_codebooks + 1``,
                mirroring upstream ``TTSSequence``).
            return_eos_frame: when True, also return the delay-aligned EOS frame
                index (or ``None`` if no EOA was sampled) so callers can trim the
                sheared codes exactly like upstream's offline decode.

        Returns:
            ``[n_frames, n_codebooks]`` int32 audio codes (delay pattern intact),
            optionally followed by the ``eos_frame`` index.
        """
        if input_ids.ndim == 2:
            input_ids = input_ids[None]
        input_ids = input_ids.astype(mx.int32)

        cache = self.backbone.make_cache()
        # softcap is already applied inside compute_logits; the sampler must not
        # re-apply it, so pass softcap=0 here.
        greedy = temperature <= 0.0 or top_k == 1
        sampler = make_codebook_sampler(
            temperature=0.0 if greedy else temperature,
            top_k=top_k,
            top_p=top_p,
            softcap=0.0,
        )

        # Prefill the prompt EXCEPT its last row as a block (this only needs to
        # populate the KV cache), then feed the last prompt row as a single-token
        # step to read the first prediction. Reading the last position straight
        # out of a long multi-token masked forward is unreliable on the MLX
        # Metal SDPA path (the final query row can be corrupted), whereas the
        # single-token decode path is exact and matches the reference; the cache
        # keys written by the block prefill are themselves correct.
        if input_ids.shape[1] > 1:
            self.backbone(input_ids[:, :-1, :], cache=cache)
        hidden = self.backbone(input_ids[:, -1:, :], cache=cache)
        last_hidden = hidden[:, -1:, :]

        frames: List[mx.array] = []
        eos_frame: Optional[int] = None
        eos_countdown = -1
        for step in range(max_frames):
            logits = self.backbone.compute_logits(last_hidden)  # [1, 1, C, V]
            codes = sampler(logits[:, 0])  # [1, C]
            codes = codes[0].astype(mx.int32)  # [C]
            frames.append(codes)

            if stop_on_eoa and eos_frame is None:
                # Delay-aware EOS: codebook j is delayed by j frames, so the
                # aligned EOS frame is the current step minus the highest EOA
                # codebook column (mirrors TTSSequence._check_eos).
                codes_np = np.asarray(codes)  # one host sync
                eoa_cols = [c for c, v in enumerate(codes_np) if int(v) == self.eoa_id]
                if eoa_cols:
                    eos_frame = max(0, step - max(eoa_cols))
                    eos_countdown = self.n_codebooks + 1
            if eos_countdown > 0:
                eos_countdown -= 1
                if eos_countdown == 0:
                    break

            row = self._next_input_row(codes)
            hidden = self.backbone(row, cache=cache)
            last_hidden = hidden[:, -1:, :]
            mx.eval(last_hidden)

        codes_out = mx.stack(frames, axis=0)
        if return_eos_frame:
            return codes_out, eos_frame
        return codes_out

    @staticmethod
    def shear_up(codes: mx.array, pad_id: int) -> mx.array:
        """Remove the delay pattern: column ``j`` shifted up by ``j`` rows.

        ``codes`` is ``[H, W]`` (frames x codebooks). Mirrors upstream
        ``vocoder.shear_up``.
        """
        H, W = codes.shape[-2], codes.shape[-1]
        out = mx.full(codes.shape, pad_id, dtype=codes.dtype)
        for j in range(W):
            if H > j:
                out[..., : H - j, j] = codes[..., j:, j]
        return out

    def decode_audio(
        self, codes: mx.array, eos_frame: Optional[int] = None
    ) -> mx.array:
        """De-shear codes and DAC-decode to a waveform.

        Mirrors upstream ``tts/llm.py`` offline decode: ``shear_up`` first, then
        (when EOS was detected) truncate to ``eos_frame`` aligned frames before
        DAC decode so the EOS frame and the post-EOS delay/countdown tail are
        dropped.

        Args:
            codes: ``[H, n_codebooks]`` raw (delayed) audio codes.
            eos_frame: delay-aligned EOS frame index; ``None`` keeps all frames.

        Returns:
            ``[samples]`` float32 waveform at 44.1 kHz (empty when no frames).
        """
        sheared = self.shear_up(codes, self.audio_pad_id)
        if eos_frame is not None:
            sheared = sheared[: max(0, eos_frame)]
        if sheared.shape[0] == 0:
            return mx.zeros((0,), dtype=mx.float32)
        # Clamp to the valid DAC codebook range (drop eoa/pad placeholders).
        sheared = mx.clip(sheared, 0, self.config.codebook_size - 1)
        # DAC expects [B, n_codebooks, T].
        dac_codes = sheared.T[None].astype(mx.int32)  # [1, C, H]
        z = self.dac.quantizer.from_codes(dac_codes)[0]
        audio = self.dac.decode(z)
        return audio.reshape(-1)

    def generate(
        self,
        text: str,
        *,
        max_frames: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        decode_audio: bool = True,
        prompt_ids: Optional[mx.array] = None,
    ):
        """Generate audio from ``text`` (or a prebuilt ``prompt_ids`` tensor).

        Returns ``(audio_codes [H, C], waveform [samples] | None)``.
        """
        if prompt_ids is None:
            raise NotImplementedError(
                "Text-to-prompt conditioning (speaker/quality buckets, silence "
                "prefix) is not yet wired; pass a prebuilt `prompt_ids` "
                "[seq, frame_width] tensor (see parity_test/zonos2/check_full.py)."
            )

        codes, eos_frame = self.generate_codes(
            prompt_ids,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_eos_frame=True,
        )
        waveform = (
            self.decode_audio(codes, eos_frame=eos_frame) if decode_audio else None
        )
        return codes, waveform
