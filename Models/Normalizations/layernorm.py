import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Channel-only LayerNorm for [B, C, L]-style tensors.

    Unlike nn.LayerNorm([C, L]), this normalizes each token position
    independently across channels only (dim=1), which matches QANet/
    Transformer-style feature normalization.
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        self.num_channels = int(num_channels)
        self.eps = eps

        # Per-channel affine parameters: [C, 1], broadcast over sequence/time.
        self.weight = nn.Parameter(torch.ones(self.num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(self.num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(f"Expected tensor with shape [B, C, ...], got {tuple(x.shape)}")
        if x.size(1) != self.num_channels:
            raise ValueError(
                f"Channel mismatch: expected C={self.num_channels}, got C={x.size(1)}"
            )

        # Normalize each position independently across channels.
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
