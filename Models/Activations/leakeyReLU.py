import torch
import torch.nn as nn


class LeakyReLU(nn.Module):
    """
    Custom LeakyReLU activation: f(x) = x if x >= 0 else negative_slope * x.
    Implemented without F.leaky_relu via torch.where.

    Args:
        negative_slope: slope for negative inputs (default: 0.01).
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.negative_slope * x)
