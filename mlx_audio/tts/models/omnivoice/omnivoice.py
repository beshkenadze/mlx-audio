import math
import time
from typing import TYPE_CHECKING, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from .backbone import BackboneConfig, OmniVoiceBackbone
from .config import OmniVoiceConfig


class Model(nn.Module):
    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config

        llm_cfg = config.llm_config or {}
        self.backbone = OmniVoiceBackbone(
            BackboneConfig(
                **{
                    k: v
                    for k, v in llm_cfg.items()
                    if k in BackboneConfig.__dataclass_fields__
                }
            )
        )

        hidden = self.backbone.embed_tokens.weight.shape[-1]
        C = config.num_audio_codebook
        V = config.audio_vocab_size  # 1025 (includes mask token)

        # 8 independent embedding tables for 8 codebooks
        self.audio_embeddings: List[nn.Embedding] = [
            nn.Embedding(V, hidden) for _ in range(C)
        ]
        # 8 independent prediction heads
        self.audio_heads: List[nn.Linear] = [
            nn.Linear(hidden, V, bias=False) for _ in range(C)
        ]

    def _embed(
        self,
        input_ids: mx.array,  # [B, S] text token ids
        audio_tokens: mx.array,  # [B, T, 8] audio codebook tokens (may include MASK_ID)
    ) -> mx.array:  # [B, S+T, hidden]
        text_embeds = self.backbone.embed_tokens(input_ids)  # [B, S, H]
        # Sum embeddings across 8 codebooks (not concat)
        audio_embeds = sum(
            self.audio_embeddings[i](audio_tokens[:, :, i])
            for i in range(self.config.num_audio_codebook)
        )  # [B, T, H]
        return mx.concatenate([text_embeds, audio_embeds], axis=1)  # [B, S+T, H]

    def build_cond_embeds(
        self,
        input_ids: mx.array,  # [1, S]
        ref_tokens: Optional[mx.array] = None,  # [1, T_ref, 8]
    ) -> mx.array:  # [1, S + T_ref, D]
        """Build conditioning embedding: text + optional reference audio tokens."""
        text_embeds = self.backbone.embed_tokens(input_ids)  # [1, S, D]
        if ref_tokens is None:
            return text_embeds
        ref_embeds = sum(
            self.audio_embeddings[i](ref_tokens[:, :, i])
            for i in range(self.config.num_audio_codebook)
        )  # [1, T_ref, D]
        return mx.concatenate([text_embeds, ref_embeds], axis=1)

    def __call__(
        self,
        inputs_embeds: mx.array,  # [B, prefix_len+T, D]
        prefix_len: int,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:  # [B, T, 8, V]
        hidden = self.backbone(inputs_embeds, attention_mask)  # [B, prefix_len+T, H]
        audio_hidden = hidden[:, prefix_len:, :]  # [B, T, H]
        logits = mx.stack(
            [
                self.audio_heads[i](audio_hidden)
                for i in range(self.config.num_audio_codebook)
            ],
            axis=2,
        )  # [B, T, 8, V]
        return logits

    def generate(
        self,
        input_ids: mx.array,  # [S] 1D text tokens
        duration_s: float,
        ref_tokens: Optional[
            mx.array
        ] = None,  # [T_ref, 8] from create_voice_clone_prompt
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        temperature: float = 5.0,
    ) -> GenerationResult:
        from .generation import iterative_unmask

        T = math.ceil(duration_s * self.config.sample_rate / 320)
        input_ids_b = input_ids[None]  # [1, S]
        ref_b = ref_tokens[None] if ref_tokens is not None else None

        cond_embeds = self.build_cond_embeds(input_ids_b, ref_b)
        uncond_embeds = self.build_cond_embeds(mx.zeros_like(input_ids_b))
        # Pad uncond to match cond's prefix length with zeros
        if cond_embeds.shape[1] > uncond_embeds.shape[1]:
            pad_len = cond_embeds.shape[1] - uncond_embeds.shape[1]
            pad = mx.zeros((1, pad_len, uncond_embeds.shape[2]))
            uncond_embeds = mx.concatenate([uncond_embeds, pad], axis=1)

        start_time = time.time()
        tokens = iterative_unmask(
            self,
            cond_embeds=cond_embeds,
            uncond_embeds=uncond_embeds,
            T=T,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            temperature=temperature,
        )
        processing_time_seconds = time.time() - start_time

        return GenerationResult(
            audio=None,
            samples=0,
            sample_rate=self.config.sample_rate,
            segment_idx=0,
            token_count=T,
            audio_duration=duration_s,
            real_time_factor=0.0,
            prompt="",
            audio_samples=tokens.tolist(),
            processing_time_seconds=processing_time_seconds,
            peak_memory_usage=0,
        )

    def sanitize(self, weights: dict) -> dict:
        result = {}
        for key, value in weights.items():
            if key.startswith("lm_head."):
                continue
            elif key.startswith("model."):
                result["backbone." + key[len("model.") :]] = value
            elif key.startswith("audio_embed."):
                result["audio_embeddings." + key[len("audio_embed.") :]] = value
            elif key.startswith("audio_head."):
                result["audio_heads." + key[len("audio_head.") :]] = value
            else:
                result[key] = value
        return result

    @property
    def model_type(self) -> str:
        return self.config.model_type

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate
