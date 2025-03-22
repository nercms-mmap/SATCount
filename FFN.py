# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch.nn.functional as F
import torch
import math
from torch import nn, Tensor

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # hw1
        self.hwsize = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout10 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)

        self.activation11 = _get_activation_fn(activation)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout20 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)

        self.activation21 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape
        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=hw,
                                   key=src1,
                                   value=src1)[0]
        src = self.norm11(src)
        src = self.linear12(self.dropout10(self.activation11(self.linear11(src))))
        src = self.norm12(src)

        srcs = self.multihead_attn2(query=src2,
                                   key=src,
                                   value=src)[0]
        srcs = self.norm21(srcs)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
        srcs = self.norm22(srcs)
        return srcs.unsqueeze(0).transpose(1, 2)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
