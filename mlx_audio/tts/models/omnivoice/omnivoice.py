import mlx.nn as nn

from .config import OmniVoiceConfig


class Model(nn.Module):
    """OmniVoice: NAR discrete diffusion TTS. Implementation in progress."""

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config
