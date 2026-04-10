import torch
import torch.nn as nn
from typing import Union, List


class LayerNorm(nn.Module):
    """
    Custom LayerNorm implementation without using nn.LayerNorm or F.layer_norm.

    Normalizes over the last len(normalized_shape) dimensions of the input.
    For a tensor of shape [B, C, L] with normalized_shape=[C, L], normalization
    is computed over the [C, L] slice of each batch element independently.

    y = (x - mean) / sqrt(var + eps) * weight + bias
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = list(normalized_shape)
        self.eps = eps

        # Learnable affine parameters (same shape as normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine which dims to reduce over (the last N dims)
        n = len(self.normalized_shape)
        dims = tuple(range(-n, 0))  # e.g. (-2, -1) for a 2-D normalized_shape

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias
