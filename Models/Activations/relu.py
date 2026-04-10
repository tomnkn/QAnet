import torch
import torch.nn as nn


class ReLU(nn.Module):
    """
    Custom ReLU activation: f(x) = max(0, x).
    Implemented without F.relu via element-wise clamping.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=0.0)
