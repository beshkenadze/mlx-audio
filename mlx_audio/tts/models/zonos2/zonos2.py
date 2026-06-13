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
from mlx.utils import tree_flatten
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

        # Optional speaker-embedding projections (voice-cloning checkpoints only).
        # ECAPA x-vector [speaker_embedding_dim] -> LDA -> [speaker_lda_dim] ->
        # speaker_projection -> [dim]; injected (index-replace) at the speaker
        # token position before emb_norm. Mirrors upstream Zonos2ForCausalLM.
        if config.speaker_enabled and config.speaker_lda_dim:
            self.speaker_lda_projection = nn.Linear(
                config.speaker_embedding_dim, int(config.speaker_lda_dim), bias=True
            )
            speaker_proj_in = int(config.speaker_lda_dim)
        else:
            self.speaker_lda_projection = None
            speaker_proj_in = config.speaker_embedding_dim
        self.speaker_projection = (
            nn.Linear(speaker_proj_in, config.dim, bias=True)
            if config.speaker_enabled
            else None
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

    def project_speaker(self, speaker_emb: mx.array) -> mx.array:
        """ECAPA x-vector ``[..., speaker_embedding_dim]`` -> backbone dim ``[..., dim]``.

        Applies the (optional) LDA projection then the speaker projection, exactly
        mirroring upstream ``_forward_model``. Raises if the checkpoint carries no
        speaker projection (base, non-cloning checkpoint).
        """
        if self.speaker_projection is None:
            raise ValueError(
                "This checkpoint has no speaker_projection; load a voice-cloning "
                "checkpoint (or merge speaker.safetensors) to condition on a voice."
            )
        emb = speaker_emb
        if self.speaker_lda_projection is not None:
            emb = self.speaker_lda_projection(emb)
        return self.speaker_projection(emb)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Optional[List[Optional[KVCache]]] = None,
        mask: Optional[mx.array] = None,
        speaker_emb: Optional[mx.array] = None,
        speaker_pos: Optional[int] = None,
    ) -> mx.array:
        """Return hidden states ``[B, T, hidden]`` for ``input_ids [B, T, W]``.

        ``speaker_emb`` (already projected to ``[dim]`` / ``[1, dim]`` via
        :meth:`project_speaker`) replaces the embedded token at sequence position
        ``speaker_pos`` (index-copy, mirroring upstream ``index_copy``), before
        ``emb_norm``. Both must be given together and ``speaker_pos`` must index a
        row that is actually present in this forward call.
        """
        if cache is None:
            cache = [None] * len(self.layers)

        x = self.multi_embedder(input_ids)
        if speaker_emb is not None and speaker_pos is not None:
            # Replace (not add) the embedding at the speaker slot. x is [B, T, dim];
            # B == 1 for generation. speaker_emb is [dim] or [1, dim]. A caller that
            # asks for injection at a position outside this forward window is a bug
            # (the embedding would be silently dropped -> un-cloned voice), so fail
            # loudly rather than open.
            if not (0 <= speaker_pos < x.shape[1]):
                raise ValueError(
                    f"speaker_pos {speaker_pos} is outside the forward window "
                    f"[0, {x.shape[1]}); cannot inject the speaker embedding."
                )
            emb_row = speaker_emb.reshape(-1).astype(x.dtype)
            x[:, speaker_pos, :] = emb_row
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

    # Speaker projection checkpoint keys (bare; addressed under ``backbone.``).
    _SPEAKER_KEYS = (
        "speaker_lda_projection.weight",
        "speaker_lda_projection.bias",
        "speaker_projection.weight",
        "speaker_projection.bias",
    )

    def load_weights(self, weights, strict: bool = True):
        """Load weights into the backbone.

        ``weights`` may be a path, a list of ``(key, array)`` pairs, or a dict.
        Keys are addressed under ``backbone.`` so the checkpoint's bare
        ``multi_embedder.* / layers.* / out_norm.* / multi_output.*`` names map
        onto :class:`Zonos2Backbone`.

        Base (non-cloning) checkpoints omit the speaker projection tensors. When
        the model carries those params but the incoming weights don't, the
        model's own freshly-initialized speaker tensors are backfilled into the
        weight set so the load can stay ``strict=True`` (validating every other
        key/shape) while leaving the unused speaker projections at their init.
        Populate them later with :meth:`load_speaker_weights`.
        """
        if isinstance(weights, (str, Path)):
            weights = dict(mx.load(str(weights)).items())
        elif isinstance(weights, list):
            weights = dict(weights)

        weights = self.sanitize(weights)
        has_speaker_params = self.backbone.speaker_projection is not None
        ckpt_has_speaker = any(k in weights for k in self._SPEAKER_KEYS)
        if has_speaker_params and not ckpt_has_speaker:
            # Backfill ONLY the four speaker keys from the model's own params so
            # the strict load still catches any genuinely missing/renamed key.
            current = dict(tree_flatten(self.backbone.parameters()))
            for k in self._SPEAKER_KEYS:
                if k in current:
                    weights[k] = current[k]
        prefixed = [(f"backbone.{k}", v) for k, v in weights.items()]
        super().load_weights(prefixed, strict=strict)
        return self

    def load_speaker_weights(self, weights) -> "Model":
        """Load just the speaker projection tensors (voice-cloning conditioning).

        ``weights`` is a path / dict / list holding the bare
        ``speaker_lda_projection.{weight,bias}`` and
        ``speaker_projection.{weight,bias}`` tensors (e.g. the extracted
        ``speaker.safetensors``). They are addressed under ``backbone.`` and
        loaded non-strict (only these four keys are touched).
        """
        if self.backbone.speaker_projection is None:
            raise ValueError(
                "Model was built without speaker projections "
                "(speaker_enabled/speaker_lda_dim is unset in the config)."
            )
        if isinstance(weights, (str, Path)):
            weights = dict(mx.load(str(weights)).items())
        elif isinstance(weights, list):
            weights = dict(weights)
        selected = {k: v for k, v in weights.items() if k in self._SPEAKER_KEYS}
        missing = [k for k in self._SPEAKER_KEYS if k not in selected]
        if missing:
            raise ValueError(
                f"Incomplete speaker projection weights; missing {missing}. "
                f"All of {list(self._SPEAKER_KEYS)} are required."
            )
        # Validate shapes against the model's params (a non-strict load would
        # otherwise silently accept a transposed/wrong-dim speaker tensor).
        current = dict(tree_flatten(self.backbone.parameters()))
        for k, v in selected.items():
            expected = current[k].shape
            if tuple(v.shape) != tuple(expected):
                raise ValueError(
                    f"Speaker tensor {k!r} has shape {tuple(v.shape)}, "
                    f"expected {tuple(expected)}."
                )
        prefixed = [(f"backbone.{k}", v) for k, v in selected.items()]
        super().load_weights(prefixed, strict=False)
        return self

    @classmethod
    def from_local(
        cls,
        model_dir: Union[str, Path],
        speaker_weights: Optional[Union[str, Path]] = None,
    ) -> "Model":
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        model = cls(config)
        model.load_weights(str(model_dir / "model.safetensors"))
        if speaker_weights is not None:
            model.load_speaker_weights(str(speaker_weights))
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

    # ── speaker-conditioned prompt assembly ──────────────────────────────────────

    def _clean_background_token(self) -> int:
        """Token id of the clean speaker-background marker.

        Conditioning tokens occupy the tail of the text vocabulary in order:
        speaking-rate, quality (per feature), speaker-background (clean, noisy),
        accurate-mode. Upstream ``speaker_background_token_id(clean=True)`` resolves
        to ``base + rate + sum(quality) + 0`` where ``base = text_vocab - rate -
        sum(quality) - bg - acc``; the rate and quality terms cancel, so the clean
        marker is simply the first of the (bg + acc) trailing slots:
        ``text_vocab - bg - acc``. Computing it this way is independent of how the
        rate/quality bucket counts happen to be represented on the config.
        """
        cfg = self.config
        bg = 2 if cfg.speaker_background_token_enabled else 0
        acc = 1 if (cfg.accurate_mode_token_enabled and bg > 0) else 0
        return int(cfg.text_vocab) - bg - acc

    def with_speaker_frames(self, prompt_ids: mx.array) -> Tuple[mx.array, int]:
        """Prepend the canonical speaker slot + clean-background marker.

        Mirrors upstream ``TTSScheduler._with_speaker_frames`` for the default
        (expressive, clean-background) case: a reserved speaker slot
        ``[audio_pad×n, text_vocab]`` at position 0, then the clean speaker-
        background marker, then the original prompt. The speaker embedding is
        injected at position 0. Returns ``(new_prompt_ids, speaker_position=0)``.
        """
        if prompt_ids.ndim == 3:
            prompt_ids = prompt_ids[0]
        prompt_ids = prompt_ids.astype(mx.int32)
        width = prompt_ids.shape[-1]
        pad = int(self.audio_pad_id)
        text_vocab = int(self.text_vocab) if self.text_vocab is not None else pad

        speaker_slot = mx.full((1, width), pad, dtype=mx.int32)
        speaker_slot[0, self.n_codebooks] = text_vocab
        rows = [speaker_slot]
        if self.config.speaker_background_token_enabled:
            bg_marker = mx.full((1, width), pad, dtype=mx.int32)
            bg_marker[0, self.n_codebooks] = self._clean_background_token()
            rows.append(bg_marker)
        new_prompt = mx.concatenate(rows + [prompt_ids], axis=0)
        return new_prompt, 0

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
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 50,
        repetition_codebooks: int = 8,
        stop_on_eoa: bool = True,
        return_eos_frame: bool = False,
        speaker_embedding: Optional[mx.array] = None,
        speaker_position: Optional[int] = None,
    ):
        """Autoregressive audio-code generation from a prebuilt prompt.

        Args:
            input_ids: prompt of shape ``[seq, frame_width]`` (or
                ``[1, seq, frame_width]``) of int ids.
            max_frames: number of audio frames to generate.
            temperature/top_k/top_p/min_p: per-codebook sampling controls (greedy
                when ``temperature <= 0`` or ``top_k == 1``).
            repetition_penalty: flat factor (``>= 1``) applied to the logits of
                tokens seen in the recent window of the SAME codebook (mirrors
                upstream ``apply_repetition_penalty``). ``<= 1`` disables it.
            repetition_window: how many recent frames feed the repetition penalty.
            repetition_codebooks: only the first N codebooks get rep-penalty
                (upstream default 8 of 9).
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

        # Project the speaker x-vector once; it conditions only the prompt prefix
        # (position ``speaker_position``), so it is applied during prefill.
        speaker_emb = None
        if speaker_embedding is not None and speaker_position is not None:
            speaker_emb = self.backbone.project_speaker(
                speaker_embedding.reshape(1, -1).astype(mx.float32)
            )

        cache = self.backbone.make_cache()
        # softcap is already applied inside compute_logits; the sampler must not
        # re-apply it, so pass softcap=0 here.
        greedy = temperature <= 0.0 or top_k == 1
        rep_penalty = max(float(repetition_penalty), 1.0)
        rep_window = max(int(repetition_window), 0)
        # Repetition penalty only matters on the stochastic path (greedy stays
        # parity-exact); disable the rolling buffer entirely when greedy.
        rep_active = (not greedy) and rep_penalty > 1.0 and rep_window > 0
        sampler = make_codebook_sampler(
            temperature=0.0 if greedy else temperature,
            top_k=top_k,
            top_p=top_p,
            softcap=0.0,
            min_p=min_p,
            repetition_penalty=rep_penalty,
            repetition_codebooks=repetition_codebooks,
            codebook_size=self.config.codebook_size,
        )

        # Prefill the prompt EXCEPT its last row as a block (this only needs to
        # populate the KV cache), then feed the last prompt row as a single-token
        # step to read the first prediction. Reading the last position straight
        # out of a long multi-token masked forward is unreliable on the MLX
        # Metal SDPA path (the final query row can be corrupted), whereas the
        # single-token decode path is exact and matches the reference; the cache
        # keys written by the block prefill are themselves correct.
        prompt_len = input_ids.shape[1]
        if prompt_len > 1:
            # Speaker slot lives in the leading block; inject it there. If the slot
            # somehow falls on the final prompt row, defer to the last-row forward.
            block_pos = (
                speaker_position
                if (
                    speaker_emb is not None and (speaker_position or 0) < prompt_len - 1
                )
                else None
            )
            self.backbone(
                input_ids[:, :-1, :],
                cache=cache,
                speaker_emb=speaker_emb if block_pos is not None else None,
                speaker_pos=block_pos,
            )
            last_pos = (
                0
                if (
                    speaker_emb is not None
                    and (speaker_position or 0) == prompt_len - 1
                )
                else None
            )
            hidden = self.backbone(
                input_ids[:, -1:, :],
                cache=cache,
                speaker_emb=speaker_emb if last_pos is not None else None,
                speaker_pos=last_pos,
            )
        else:
            last_pos = 0 if (speaker_emb is not None) else None
            hidden = self.backbone(
                input_ids[:, -1:, :],
                cache=cache,
                speaker_emb=speaker_emb if last_pos is not None else None,
                speaker_pos=last_pos,
            )
        last_hidden = hidden[:, -1:, :]

        frames: List[mx.array] = []
        eos_frame: Optional[int] = None
        eos_countdown = -1
        for step in range(max_frames):
            logits = self.backbone.compute_logits(last_hidden)  # [1, 1, C, V]
            # Rolling repetition-penalty window: the last ``rep_window`` emitted
            # frames as a [1, C, window] buffer of per-codebook ids (upstream
            # caps the active window at the number of generated frames so far).
            rep_ids = None
            if rep_active and frames:
                window = frames[-rep_window:]
                # stack frames [w, C] -> transpose to [C, w] -> [1, C, w]
                rep_ids = mx.stack(window, axis=0).T[None]
            codes = sampler(logits[:, 0], rep_ids)  # [1, C]
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

    @staticmethod
    def _trim_leading_silence(
        wav: mx.array,
        *,
        threshold: float = 0.01,
        keep_samples: int = 256,
    ) -> mx.array:
        """Drop the run of leading near-silent samples (post-hoc cosmetic trim).

        The model is trained to emit a short (~0.2 s) leading-silence prefix; the
        upstream offline decode keeps it, but it reads as a dead pause at the head
        of the clip. We strip the contiguous leading region whose absolute
        amplitude stays below ``threshold``, leaving ``keep_samples`` of the
        silence as a natural lead-in so the first phoneme is not clipped. Returns
        the waveform unchanged when it never crosses the threshold (all-silent).
        """
        if wav.shape[0] == 0:
            return wav
        above = mx.abs(wav) >= threshold
        if not bool(mx.any(above)):
            return wav
        first = int(mx.argmax(above).item())  # first index at/above threshold
        start = max(0, first - keep_samples)
        return wav[start:]

    def decode_audio(
        self,
        codes: mx.array,
        eos_frame: Optional[int] = None,
        *,
        trim_leading_silence: bool = False,
    ) -> mx.array:
        """De-shear codes and DAC-decode to a waveform.

        Mirrors upstream ``tts/llm.py`` offline decode: ``shear_up`` first, then
        (when EOS was detected) truncate to ``eos_frame`` aligned frames before
        DAC decode so the EOS frame and the post-EOS delay/countdown tail are
        dropped.

        Args:
            codes: ``[H, n_codebooks]`` raw (delayed) audio codes.
            eos_frame: delay-aligned EOS frame index; ``None`` keeps all frames.
            trim_leading_silence: when True, strip the leading near-silent samples
                from the decoded waveform (post-hoc; does not alter the codes that
                feed DAC, only the returned audio).

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
        audio = self.dac.decode(z).reshape(-1)
        if trim_leading_silence:
            audio = self._trim_leading_silence(audio)
        return audio

    def generate(
        self,
        text: str,
        *,
        max_frames: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        repetition_window: int = 50,
        repetition_codebooks: int = 8,
        stop_on_eoa: bool = True,
        decode_audio: bool = True,
        trim_leading_silence: bool = False,
        prompt_ids: Optional[mx.array] = None,
        speaker_embedding: Optional[mx.array] = None,
    ):
        """Generate audio from ``text`` (or a prebuilt ``prompt_ids`` tensor).

        ``temperature``/``top_k``/``top_p``/``min_p``/``repetition_penalty`` mirror
        the upstream ``TTSSamplingParams``; pass the model defaults
        (``temperature=1.15, top_k=106, top_p=0.0, min_p=0.18,
        repetition_penalty=1.2``) for the clean, full-quality sampler.

        When ``speaker_embedding`` (a ``[2048]`` / ``[1, 2048]`` ECAPA x-vector)
        is given, the prompt is wrapped with the canonical speaker slot +
        clean-background marker and the embedding is injected at the speaker token
        position (mirrors upstream ``_with_speaker_frames`` / ``_forward_model``),
        cloning that voice.

        ``stop_on_eoa=True`` (the natural mode) lets the model finish the sentence
        and emit the end-of-audio token, then trims the delay tail at the aligned
        eos_frame — set a generous ``max_frames`` so the full utterance fits.

        Returns ``(audio_codes [H, C], waveform [samples] | None)``.
        """
        if prompt_ids is None:
            raise NotImplementedError(
                "Text-to-prompt conditioning (speaker/quality buckets, silence "
                "prefix) is not yet wired; pass a prebuilt `prompt_ids` "
                "[seq, frame_width] tensor (see parity_test/zonos2/check_full.py)."
            )

        speaker_pos: Optional[int] = None
        if speaker_embedding is not None:
            prompt_ids, speaker_pos = self.with_speaker_frames(prompt_ids)

        codes, eos_frame = self.generate_codes(
            prompt_ids,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            repetition_codebooks=repetition_codebooks,
            stop_on_eoa=stop_on_eoa,
            return_eos_frame=True,
            speaker_embedding=speaker_embedding,
            speaker_position=speaker_pos,
        )
        waveform = (
            self.decode_audio(
                codes,
                eos_frame=eos_frame,
                trim_leading_silence=trim_leading_silence,
            )
            if decode_audio
            else None
        )
        return codes, waveform
