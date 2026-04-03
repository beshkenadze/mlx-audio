import mlx.core as mx
import mlx.nn as nn

from .config import HiggsAudioConfig


class HiggsAudioTokenizer(nn.Module):
    """
    HiggsAudioV2 tokenizer: DAC-based decoder for OmniVoice acoustic tokens.

    NOTE: Weight loading not yet implemented. This skeleton defines the
    decode() interface. Full implementation requires inspecting safetensors
    weight keys from k2-fsa/OmniVoice audio_tokenizer/model.safetensors.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config

    def decode(self, tokens: mx.array) -> mx.array:
        """
        Decode multi-codebook tokens to waveform.

        Args:
            tokens: mx.array of shape [T, num_codebooks] int32

        Returns:
            mx.array of shape [T * downsample_factor] float32 waveform

        NOTE: Not yet implemented — raises NotImplementedError.
        Full implementation blocked by weight inspection (see docs/omnivoice-porting-research.md §9).
        """
        raise NotImplementedError(
            "HiggsAudioV2 decoder not yet implemented. "
            "Requires weight inspection of audio_tokenizer/model.safetensors."
        )

    @classmethod
    def from_pretrained(cls, model_path: str) -> "HiggsAudioTokenizer":
        """Load from HuggingFace model directory. Not yet implemented."""
        raise NotImplementedError("Weight loading not yet implemented.")
