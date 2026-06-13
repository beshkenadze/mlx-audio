from dataclasses import dataclass, field


@dataclass
class EcapaTdnnConfig:
    input_size: int = 60
    channels: int = 1024
    embed_dim: int = 256
    kernel_sizes: list[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    attention_channels: int = 128
    res2net_scale: int = 8
    se_channels: int = 128
    global_context: bool = False
    # Conv "same" padding mode for TDNN blocks. SpeechBrain ECAPA (LID/spark)
    # uses zero padding; the Qwen3-TTS / ZONOS2 ECAPA checkpoint uses reflect
    # padding (``padding="same", padding_mode="reflect"``). Only kernels > 1 are
    # affected; the k=1 SE/ASP/fc convs are mode-independent.
    conv_padding_mode: str = "zeros"
