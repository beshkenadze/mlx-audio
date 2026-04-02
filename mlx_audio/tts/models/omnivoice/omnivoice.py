from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

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

    def __call__(
        self,
        input_ids: mx.array,  # [B, S]
        audio_tokens: mx.array,  # [B, T, 8]
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:  # [B, T, 8, V]
        inputs_embeds = self._embed(input_ids, audio_tokens)  # [B, S+T, H]
        hidden = self.backbone(inputs_embeds, attention_mask)  # [B, S+T, H]
        S = input_ids.shape[1]
        audio_hidden = hidden[:, S:, :]  # [B, T, H]

        logits = mx.stack(
            [
                self.audio_heads[i](audio_hidden)
                for i in range(self.config.num_audio_codebook)
            ],
            axis=2,
        )  # [B, T, 8, V]
        return logits

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
