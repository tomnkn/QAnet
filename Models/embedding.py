import torch
import torch.nn as nn

from .conv import DepthwiseSeparableConv
from .dropout import Dropout
from .Activations import get_activation


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int, act_name: str = "relu"):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.act = get_activation(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.act(self.linear[i](x))
            x = gate * nonlinear + (1.0 - gate) * x
        return x.transpose(1, 2)


class Embedding(nn.Module):
    def __init__(self, d_word: int, d_char: int, dropout: float, dropout_char: float, init_name: str = "kaiming", act_name: str = "relu"):
        super().__init__()
        self.drop = Dropout(dropout)
        self.drop_char = Dropout(dropout_char)
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2, init_name=init_name)
        self.high = Highway(2, d_word + d_char, act_name=act_name)
        self.act = get_activation(act_name)

    def forward(self, ch_emb: torch.Tensor, wd_emb: torch.Tensor) -> torch.Tensor:
        # ch_emb: [B, L, char_len, d_char]
        # wd_emb: [B, L, d_word]
        ch_emb = ch_emb.permute(0, 3, 1, 2)  # [B, d_char, L, char_len]
        ch_emb = self.drop_char(ch_emb)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = self.act(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)  # [B, d_char, L]

        wd_emb = self.drop(wd_emb)
        wd_emb = wd_emb.transpose(1, 2)  # [B, d_word, L]

        emb = torch.cat([ch_emb, wd_emb], dim=1)  # [B, d_char+d_word, L]
        emb = self.high(emb)
        return emb
