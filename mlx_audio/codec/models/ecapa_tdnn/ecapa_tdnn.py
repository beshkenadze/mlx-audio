import mlx.core as mx
import mlx.nn as nn

from .config import EcapaTdnnConfig


def _reflect_pad_time(x: mx.array, pad: int) -> mx.array:
    """Reflect-pad an ``[B, L, C]`` tensor by ``pad`` frames on each side of L.

    Matches ``torch.nn.functional.pad(..., mode="reflect")`` (the boundary frame
    itself is excluded from the reflection), including its ``pad < L`` requirement:
    a too-short input raises rather than silently under-padding (which would yield
    a wrong-length, non-reflected tensor that diverges from torch without error).
    """
    if pad <= 0:
        return x
    if x.shape[1] <= pad:
        raise ValueError(
            f"sequence too short for reflect pad: length {x.shape[1]} <= pad {pad}"
        )
    left = x[:, 1 : pad + 1, :][:, ::-1, :]
    right = x[:, -(pad + 1) : -1, :][:, ::-1, :]
    return mx.concatenate([left, x, right], axis=1)


class TDNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if padding_mode not in ("zeros", "reflect"):
            raise ValueError(
                f"padding_mode must be 'zeros' or 'reflect', got {padding_mode!r}"
            )
        self.same_pad = (kernel_size - 1) * dilation // 2
        self.padding_mode = padding_mode
        # Reflect "same" padding is applied manually (MLX Conv1d only zero-pads),
        # so the conv itself pads internally only in the default "zeros" mode.
        conv_pad = 0 if padding_mode == "reflect" else self.same_pad
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=conv_pad,
            dilation=dilation,
            bias=True,
        )
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if self.padding_mode == "reflect":
            x = _reflect_pad_time(x, self.same_pad)
        return self.norm(nn.relu(self.conv(x)))


class Res2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by scale ({scale})"
            )
        self.scale = scale
        hidden = channels // scale
        self.blocks = [
            TDNNBlock(hidden, hidden, kernel_size, dilation, padding_mode)
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        chunks = mx.split(x, self.scale, axis=-1)
        y = [chunks[0]]
        for i, block in enumerate(self.blocks):
            inp = chunks[i + 1] + y[-1] if i > 0 else chunks[i + 1]
            y.append(block(inp))
        return mx.concatenate(y, axis=-1)


class SEBlock(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, bottleneck, 1)
        self.conv2 = nn.Conv1d(bottleneck, in_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        s = mx.mean(x, axis=1, keepdims=True)
        s = nn.relu(self.conv1(s))
        s = mx.sigmoid(self.conv2(s))
        return x * s


class SERes2NetBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        res2net_scale: int = 8,
        se_channels: int = 128,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.tdnn1 = TDNNBlock(channels, channels, 1, padding_mode=padding_mode)
        self.res2net_block = Res2NetBlock(
            channels, kernel_size, dilation, res2net_scale, padding_mode
        )
        self.tdnn2 = TDNNBlock(channels, channels, 1, padding_mode=padding_mode)
        self.se_block = SEBlock(channels, se_channels)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.tdnn1(x)
        out = self.res2net_block(out)
        out = self.tdnn2(out)
        out = self.se_block(out)
        return out + x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(
        self,
        channels: int,
        attention_channels: int = 128,
        global_context: bool = False,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.global_context = global_context
        tdnn_in = channels * 3 if global_context else channels
        self.tdnn = TDNNBlock(tdnn_in, attention_channels, 1, padding_mode=padding_mode)
        self.conv = nn.Conv1d(attention_channels, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        if self.global_context:
            m = mx.mean(x, axis=1, keepdims=True)
            v = mx.var(x, axis=1, keepdims=True)
            s = mx.sqrt(v + 1e-9)
            m_exp = mx.broadcast_to(m, x.shape)
            s_exp = mx.broadcast_to(s, x.shape)
            attn_in = mx.concatenate([x, m_exp, s_exp], axis=-1)
        else:
            attn_in = x

        attn = self.tdnn(attn_in)
        attn = mx.tanh(attn)
        attn = self.conv(attn)
        attn = mx.softmax(attn, axis=1)

        weighted_mean = mx.sum(attn * x, axis=1)
        weighted_var = mx.sum(attn * (x * x), axis=1) - weighted_mean * weighted_mean
        weighted_std = mx.sqrt(mx.maximum(weighted_var, 1e-9))

        return mx.concatenate([weighted_mean, weighted_std], axis=-1)


class EcapaTdnnBackbone(nn.Module):
    def __init__(self, config: EcapaTdnnConfig):
        super().__init__()
        ch = config.channels
        pm = config.conv_padding_mode

        self.block0 = TDNNBlock(
            config.input_size, ch, config.kernel_sizes[0], padding_mode=pm
        )
        self.block1 = SERes2NetBlock(
            ch,
            config.kernel_sizes[1],
            config.dilations[1],
            config.res2net_scale,
            config.se_channels,
            pm,
        )
        self.block2 = SERes2NetBlock(
            ch,
            config.kernel_sizes[2],
            config.dilations[2],
            config.res2net_scale,
            config.se_channels,
            pm,
        )
        self.block3 = SERes2NetBlock(
            ch,
            config.kernel_sizes[3],
            config.dilations[3],
            config.res2net_scale,
            config.se_channels,
            pm,
        )
        self.blocks = [self.block1, self.block2, self.block3]

        self.mfa = TDNNBlock(ch * 3, ch * 3, config.kernel_sizes[4], padding_mode=pm)
        self.asp = AttentiveStatisticsPooling(
            ch * 3,
            config.attention_channels,
            config.global_context,
            pm,
        )
        self.asp_bn = nn.BatchNorm(ch * 6)
        self.fc = nn.Conv1d(ch * 6, config.embed_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.block0(x)
        xl = []
        out = self.block1(out)
        xl.append(out)
        out = self.block2(out)
        xl.append(out)
        out = self.block3(out)
        xl.append(out)

        out = mx.concatenate(xl, axis=-1)
        out = self.mfa(out)
        out = self.asp(out)
        out = self.asp_bn(out)
        out = mx.expand_dims(out, axis=1)
        out = self.fc(out)
        return out.squeeze(axis=1)
