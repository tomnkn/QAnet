import torch
import torch.nn as nn


class Dropout(nn.Module):
    """Inverted dropout: keeps each element with probability (1-p) and
    scales surviving elements by 1/(1-p) so the expected value is preserved."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p))
        return x * mask / (1.0 - self.p)
