import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import mask_logits
from .Initializations import uniform_


class Pointer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3.0 / (2.0 * d_model)
        uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1: torch.Tensor, M2: torch.Tensor, M3: torch.Tensor, mask: torch.Tensor):
        X1 = torch.cat([M1, M2], dim=1)  # [B, 2C, L]
        X2 = torch.cat([M1, M3], dim=1)  # [B, 2C, L]
        Y1 = torch.matmul(X1.transpose(1, 2), self.w1)  # [B, L]
        Y2 = torch.matmul(X2.transpose(1, 2), self.w2)  # [B, L]
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2
