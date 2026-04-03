import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import HiggsAudioConfig
from .dac import AcousticDecoder, AcousticEncoder, ResidualVectorQuantizer


class HiggsAudioTokenizer(nn.Module):
    """
    HiggsAudioV2 acoustic tokenizer (Branch A: acoustic-only, no HuBERT).

    Decode path (tokens -> waveform): quantizer -> fc2 -> acoustic_decoder
    Encode path (waveform -> tokens): acoustic_encoder -> _enc_proj -> quantizer

    Note: encode path uses a randomly-inited projection (_enc_proj) that is not
    in the checkpoint. Encode quality requires real weights + full model.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        self.acoustic_encoder = AcousticEncoder()
        self.quantizer = ResidualVectorQuantizer()
        self.acoustic_decoder = AcousticDecoder()
        # Decode path: quantizer (1024-dim) -> fc2 -> decoder (256-dim)
        # fc2.weight shape in checkpoint: [256, 1024] = Linear(1024->256, bias=False)
        self.fc2 = nn.Linear(1024, 256, bias=False)
        # Encode bridge: encoder output (256-dim) -> quantizer input (1024-dim)
        # Not in checkpoint; randomly inited. Encode path is approximate.
        self._enc_proj = nn.Linear(256, 1024, bias=False)

    def decode(self, tokens: mx.array) -> mx.array:
        """
        tokens: [T, 8] or [B, T, 8] int32
        Returns: [T*960] (1D) if 2D input, or [B, T*960, 1] if 3D input
        """
        squeeze = tokens.ndim == 2
        if squeeze:
            tokens = tokens[None]  # [1, T, 8]
        z = self.quantizer.decode(tokens)  # [B, T, 1024]
        z = self.fc2(z)  # [B, T, 256]
        wav = self.acoustic_decoder(z)  # [B, T*960, 1]
        if squeeze:
            return wav[0, :, 0]  # [T*960]
        return wav  # [B, T*960, 1]

    def encode(self, waveform: mx.array) -> mx.array:
        """
        waveform: [B, T, 1] float32 at 24kHz
        Returns: [B, T//960, 8] int32 codebook tokens

        WARNING: Uses randomly-inited _enc_proj; results are not meaningful
        until real weights are loaded via from_pretrained().
        """
        z = self.acoustic_encoder(waveform)  # [B, T//960, 256]
        z = self._enc_proj(z)  # [B, T//960, 1024]
        return self.quantizer.encode(z)  # [B, T//960, 8]

    def sanitize(self, weights: dict) -> dict:
        """Filter checkpoint keys to only acoustic path (no semantic model)."""
        keep_prefixes = ("acoustic_encoder.", "acoustic_decoder.", "quantizer.", "fc2.")
        drop_suffixes = (".embed_avg", ".cluster_size", ".inited")
        result = {}
        for k, v in weights.items():
            if any(k.startswith(p) for p in keep_prefixes):
                if not any(k.endswith(s) for s in drop_suffixes):
                    result[k] = v
        return result

    @classmethod
    def from_pretrained(cls, model_path: str) -> "HiggsAudioTokenizer":
        """
        Load from k2-fsa/OmniVoice local directory.
        Expects: <model_path>/audio_tokenizer/config.json
                 <model_path>/audio_tokenizer/model.safetensors
        """
        config_path = Path(model_path) / "audio_tokenizer" / "config.json"
        weights_path = Path(model_path) / "audio_tokenizer" / "model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        config = HiggsAudioConfig.from_dict(json.loads(config_path.read_text()))
        inst = cls(config)
        raw = mx.load(str(weights_path))
        sanitized = inst.sanitize(raw)
        inst.load_weights(list(sanitized.items()))
        mx.eval(inst.parameters())
        return inst
