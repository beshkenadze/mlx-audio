"""Configuration for the ZONOS2 MLX port.

Mirrors the field semantics of the upstream ``zonos2.models.config.ModelConfig``
derivation (``from_zonos2_config``) but keeps the raw ``params.json`` field names
as the source of truth. Derived quantities (``intermediate_size``, ``vocab_size``,
MoE-layer predicate, per-layer top-k) are exposed as properties / methods so every
sub-module agrees on a single definition.

Reference: https://github.com/Zyphra/ZONOS2 -> python/zonos2/models/config.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Tuple


def _normalize_special_topk_layers(
    special: Optional[Dict[Any, Any]],
) -> Optional[Dict[int, int]]:
    """Coerce the ``{layer: topk}`` mapping to int keys/values (JSON gives str keys)."""
    if special is None:
        return None
    normalized: Dict[int, int] = {}
    for layer_idx, topk in special.items():
        layer_idx = int(layer_idx)
        topk = int(topk)
        if topk < 1:
            raise ValueError(
                f"special_topk_layers[{layer_idx}] must be >= 1, got {topk}"
            )
        normalized[layer_idx] = topk
    return normalized


@dataclass
class ZONOS2Config:
    """ZONOS2 backbone + TTS configuration (defaults match the released checkpoint)."""

    # --- transformer backbone (raw params.json) ---
    n_layers: int = 28
    dim: int = 2048
    head_dim: int = 128
    n_heads: Optional[int] = None  # None -> dim // head_dim
    n_kv_heads: int = 4
    ffn_dim_multiplier: float = 1.5
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seqlen: int = 6144
    hidden_act: str = "silu"

    # --- multi-codebook / TTS ---
    n_codebooks: int = 9
    codebook_size: int = 1024
    eoa_id: int = 1024
    audio_pad_id: int = 1025
    text_vocab: Optional[int] = 519
    loss_softcap: float = 15.0

    # --- speaker conditioning ---
    speaker_enabled: bool = True
    speaker_embedding_dim: int = 2048
    speaker_lda_dim: Optional[int] = 1024
    speaker_background_token_enabled: bool = True
    accurate_mode_token_enabled: bool = True
    speaking_rate_num_buckets: int = 8
    speaking_rate_buckets: Tuple[str, ...] = ()
    quality_num_buckets: int = 60
    quality_features: Tuple[str, ...] = ()
    quality_buckets: Optional[Dict[str, Tuple[str, ...]]] = None
    quality_dropout: Optional[Dict[str, float]] = None

    # --- MoE ---
    moe_impl: str = "sonic"
    moe_n_experts: int = 16
    moe_router_topk: int = 1
    special_topk_layers: Optional[Dict[int, int]] = field(default=None)
    moe_router_dim: int = 128
    moe_start_from_layer: int = 3
    moe_end_from_layer: int = 1
    moe_balancing_strategy: str = "legacy"

    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        self.special_topk_layers = _normalize_special_topk_layers(
            self.special_topk_layers
        )
        if isinstance(self.speaking_rate_buckets, list):
            self.speaking_rate_buckets = tuple(self.speaking_rate_buckets)
        if isinstance(self.quality_features, list):
            self.quality_features = tuple(self.quality_features)

    # --- derived quantities (single source of truth) ---
    @property
    def num_qo_heads(self) -> int:
        return self.n_heads if self.n_heads is not None else self.dim // self.head_dim

    @property
    def intermediate_size(self) -> int:
        """FFN inner dim: round(ffn_dim_multiplier * dim) up to ``multiple_of``.

        NOTE: ZONOS2 does NOT use the llama 2/3 rule. For the release this is
        ``256 * ceil(1.5*2048 / 256) = 3072``.
        """
        ffn = int(self.ffn_dim_multiplier * self.dim)
        return self.multiple_of * ((ffn + self.multiple_of - 1) // self.multiple_of)

    @property
    def moe_intermediate_size(self) -> int:
        return self.intermediate_size if self.moe_n_experts > 1 else 0

    @property
    def audio_vocab(self) -> int:
        """Per-codebook audio vocab: codebook_size + eoa + pad."""
        return self.codebook_size + 2

    @property
    def vocab_size(self) -> int:
        """Total output vocab = n_codebooks*(codebook_size+2) + (text_vocab+1)."""
        v = self.n_codebooks * self.audio_vocab
        if self.text_vocab is not None:
            v += self.text_vocab + 1
        return v

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Layers ``moe_start_from_layer <= i < n_layers - moe_end_from_layer`` are MoE."""
        if self.moe_n_experts <= 1:
            return False
        if layer_idx < self.moe_start_from_layer:
            return False
        if layer_idx >= self.n_layers - self.moe_end_from_layer:
            return False
        return True

    def num_experts_per_tok(self, layer_idx: int) -> int:
        """Default top-k, overridden per layer by ``special_topk_layers`` (e.g. {26: 2})."""
        default = self.moe_router_topk if self.moe_router_topk > 0 else 1
        if self.special_topk_layers is None:
            return default
        return int(self.special_topk_layers.get(layer_idx, default))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ZONOS2Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
