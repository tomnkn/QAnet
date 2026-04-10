import torch
import torch.nn as nn

from .Initializations import initializations, constant_


class Conv1d(nn.Module):
    """
    Custom 1-D convolution from scratch — no F.conv1d / nn.Conv1d.

    Algorithm:
      1. Manual zero-padding via torch.cat.
      2. Sliding-window extraction with tensor.unfold() → [B, C_in, L_out, k].
      3. Grouped multiply-accumulate via torch.einsum.

    Weight shape: [out_channels, in_channels // groups, kernel_size]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, L]
        B, C_in, _ = x.shape

        # 1. Manual zero-padding along the length dimension
        if self.padding > 0:
            p = self.padding
            pad = x.new_zeros(B, C_in, p)
            x = torch.cat([pad, x, pad], dim=2)  # [B, C_in, L + 2p]

        # 2. Sliding window: tensor.unfold(dim, size, step) → [B, C_in, L_out, k]
        L_out = x.size(2) - self.kernel_size + 1
        x_unf = x.unfold(2, self.kernel_size, 1)  # [B, C_in, L_out, k]

        # 3. Grouped multiply-accumulate
        G       = self.groups
        C_in_g  = C_in // G
        C_out_g = self.out_channels // G

        # Reshape into groups: [B, G, C_in_g, L_out, k]
        x_unf = x_unf.contiguous().view(B, G, C_in_g, L_out, self.kernel_size)
        # Weight by group: [G, C_out_g, C_in_g, k]
        w = self.weight.view(G, C_out_g, C_in_g, self.kernel_size)

        # out[b,g,o,l] = Σ_{i,k} x_unf[b,g,i,l,k] * w[g,o,i,k]
        out = torch.einsum('bgilk,goik->bgol', x_unf, w)  # [B, G, C_out_g, L_out]
        out = out.reshape(B, self.out_channels, L_out)

        if self.bias is not None:
            out = out + self.bias.view(1, self.out_channels, 1)

        return out


class Conv2d(nn.Module):
    """
    Custom 2-D convolution from scratch — no F.conv2d / nn.Conv2d.

    Algorithm:
      1. Manual zero-padding via torch.cat (height then width).
      2. Sliding-window extraction with two tensor.unfold() calls
         → [B, C_in, H_out, W_out, k, k].
      3. Grouped multiply-accumulate via torch.einsum.

    Weight shape: [out_channels, in_channels // groups, kernel_size, kernel_size]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        B, C_in, H, W = x.shape
        k = self.kernel_size

        # 1. Manual 2-D zero-padding
        if self.padding > 0:
            p = self.padding
            pad_h = x.new_zeros(B, C_in, p, W)
            x = torch.cat([pad_h, x, pad_h], dim=2)       # [B, C_in, H+2p, W]
            pad_w = x.new_zeros(B, C_in, x.size(2), p)
            x = torch.cat([pad_w, x, pad_w], dim=3)       # [B, C_in, H+2p, W+2p]

        # 2. Sliding window along height then width
        H_out = x.size(2) - k + 1
        W_out = x.size(3) - k + 1
        # unfold dim=2 (height), then dim=3 (width)
        x_unf = x.unfold(2, k, 1).unfold(3, k, 1)        # [B, C_in, H_out, W_out, k, k]

        # 3. Grouped multiply-accumulate
        G       = self.groups
        C_in_g  = C_in // G
        C_out_g = self.out_channels // G

        # Reshape into groups: [B, G, C_in_g, H_out, W_out, k, k]
        x_unf = x_unf.contiguous().view(B, G, C_in_g, H_out, W_out, k, k)
        # Weight by group: [G, C_out_g, C_in_g, k, k]
        w = self.weight.view(G, C_out_g, C_in_g, k, k)

        # out[b,g,o,h,w] = Σ_{i,p,q} x_unf[b,g,i,h,w,p,q] * w[g,o,i,p,q]
        out = torch.einsum('bgihwpq,goipq->bgohw', x_unf, w)  # [B, G, C_out_g, H_out, W_out]
        out = out.reshape(B, self.out_channels, H_out, W_out)

        if self.bias is not None:
            out = out + self.bias.view(1, self.out_channels, 1, 1)

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dim: int = 1, bias: bool = True, init_name: str = "kaiming"):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = Conv1d(in_ch, in_ch, k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = Conv1d(in_ch, out_ch, 1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = Conv2d(in_ch, in_ch, k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = Conv2d(in_ch, out_ch, 1, padding=0, bias=bias)
        else:
            raise ValueError("dim must be 1 or 2")

        init_fn = initializations[init_name]
        init_fn(self.depthwise_conv.weight)
        if self.depthwise_conv.bias is not None:
            constant_(self.depthwise_conv.bias, 0.0)

        init_fn(self.pointwise_conv.weight)
        if self.pointwise_conv.bias is not None:
            constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))
