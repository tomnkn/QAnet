import torch
import torch.nn as nn

from .conv import Conv1d
from .embedding import Embedding
from .encoder import EncoderBlock
from .attention import CQAttention
from .heads import Pointer
from .Initializations import initializations, constant_


class QANet(nn.Module):
    """
    Args-driven QANet. No config.py dependency.
    Required args fields:
      - d_model, num_heads, glove_dim, char_dim, dropout, dropout_char
      - para_limit, ques_limit
      - pretrained_char (optional bool, default False)
            - freeze_word (optional bool, default True)
    """
    def __init__(self, word_mat, char_mat, args):
        super().__init__()
        d_model = int(args.d_model)
        num_heads = int(args.num_heads)
        d_word = int(args.glove_dim)
        d_char = int(args.char_dim)
        dropout = float(args.dropout)
        dropout_char = float(args.dropout_char)
        len_c = int(args.para_limit)
        len_q = int(args.ques_limit)
        pretrained_char = bool(getattr(args, "pretrained_char", False))
        freeze_word = bool(getattr(args, "freeze_word", True))
        init_name   = str(getattr(args, "init_name",   "kaiming"))
        act_name    = str(getattr(args, "activation",  "relu"))
        norm_name   = str(getattr(args, "norm_name",   "layer_norm"))
        norm_groups = int(getattr(args, "norm_groups", 8))

        self.char_emb = nn.Embedding.from_pretrained(
            torch.tensor(char_mat, dtype=torch.float32),
            freeze=pretrained_char
        )
        self.word_emb = nn.Embedding.from_pretrained(
            torch.tensor(word_mat, dtype=torch.float32),
            freeze=freeze_word
        )

        self.emb = Embedding(d_word, d_char, dropout, dropout_char, init_name=init_name, act_name=act_name)
        # Shared 1x1 projection for both context and question streams.
        self.conv = Conv1d(d_word + d_char, d_model, 1)
        initializations[init_name](self.conv.weight)
        if self.conv.bias is not None:
            constant_(self.conv.bias, 0.0)

        # Shared embedding encoder keeps both streams in the same feature space.
        self.emb_enc = EncoderBlock(
            d_model,
            num_heads,
            dropout,
            conv_num=4,
            k=7,
            length=max(len_c, len_q),
            init_name=init_name,
            act_name=act_name,
            norm_name=norm_name,
            norm_groups=norm_groups,
        )

        self.cq_att = CQAttention(d_model, dropout)
        self.cq_resizer = Conv1d(d_model * 4, d_model, 1)
        initializations[init_name](self.cq_resizer.weight)
        if self.cq_resizer.bias is not None:
            constant_(self.cq_resizer.bias, 0.0)

        self.model_enc_blks_1 = nn.ModuleList([
            EncoderBlock(
                d_model,
                num_heads,
                dropout,
                conv_num=2,
                k=5,
                length=len_c,
                init_name=init_name,
                act_name=act_name,
                norm_name=norm_name,
                norm_groups=norm_groups,
            )
            for _ in range(7)
        ])

        self.model_enc_blks_2 = nn.ModuleList([
            EncoderBlock(
                d_model,
                num_heads,
                dropout,
                conv_num=2,
                k=5,
                length=len_c,
                init_name=init_name,
                act_name=act_name,
                norm_name=norm_name,
                norm_groups=norm_groups,
            )
            for _ in range(7)
        ])

        self.model_enc_blks_3 = nn.ModuleList([
            EncoderBlock(
                d_model,
                num_heads,
                dropout,
                conv_num=2,
                k=5,
                length=len_c,
                init_name=init_name,
                act_name=act_name,
                norm_name=norm_name,
                norm_groups=norm_groups,
            )
            for _ in range(7)
        ])

        self.out = Pointer(d_model)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = (Cwid == 0)  # True means PAD
        qmask = (Qwid == 0)

        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)

        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        C = self.conv(C)
        Q = self.conv(Q)

        Ce = self.emb_enc(C, cmask)
        Qe = self.emb_enc(Q, qmask)

        X = self.cq_att(Ce, Qe, cmask, qmask)

        M1 = self.cq_resizer(X)
        for enc in self.model_enc_blks_1:
            M1 = enc(M1, cmask)

        M2 = M1
        for enc in self.model_enc_blks_2:
            M2 = enc(M2, cmask)

        M3 = M2
        for enc in self.model_enc_blks_3:
            M3 = enc(M3, cmask)

        p1, p2 = self.out(M1, M2, M3, cmask)
        return p1, p2
