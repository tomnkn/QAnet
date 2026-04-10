import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropout import Dropout
from .encoder import mask_logits
from .Initializations import uniform_


class CQAttention(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.drop = Dropout(dropout)
        w = torch.empty(d_model * 3)
        lim = 1.0 / d_model
        uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C: torch.Tensor, Q: torch.Tensor, cmask: torch.Tensor, qmask: torch.Tensor) -> torch.Tensor:
        # C: [B, C, Lc], Q: [B, C, Lq]
        C = C.transpose(1, 2)  # [B, Lc, C]
        Q = Q.transpose(1, 2)  # [B, Lq, C]

        cmask = cmask.unsqueeze(2)  # [B, Lc, 1]
        qmask = qmask.unsqueeze(1)  # [B, 1, Lq]

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))  # [B, Lc, Lq, C]
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = Ct * Qt
        S = torch.cat([Ct, Qt, CQ], dim=3)  # [B, Lc, Lq, 3C]
        S = torch.matmul(S, self.w)  # [B, Lc, Lq]

        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)

        out = torch.cat([C, A, C * A, C * B], dim=2)  # [B, Lc, 4C]
        out = self.drop(out)
        return out.transpose(1, 2)  # [B, 4C, Lc]
