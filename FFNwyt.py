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

# class FeatureFusionLayerwyt_scale15(nn.Module):
#
#     def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
#                  activation="relu"):
#         super().__init__()
#         # hw1
#         self.hwsize = nn.Sequential(
#             nn.Linear(2, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 64)
#         )
#
#         self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#
#         self.linear11 = nn.Linear(d_model, dim_feedforward)
#         self.dropout10 = nn.Dropout(dropout)
#         self.linear12 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm11 = nn.LayerNorm(d_model)
#         self.norm12 = nn.LayerNorm(d_model)
#         self.dropout11 = nn.Dropout(dropout)
#         self.dropout12 = nn.Dropout(dropout)
#
#         self.activation11 = _get_activation_fn(activation)
#
#         self.linear21 = nn.Linear(d_model, dim_feedforward)
#         self.dropout20 = nn.Dropout(dropout)
#         self.linear22 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm21 = nn.LayerNorm(d_model)
#         self.norm22 = nn.LayerNorm(d_model)
#         self.dropout21 = nn.Dropout(dropout)
#         self.dropout22 = nn.Dropout(dropout)
#
#         self.activation21 = _get_activation_fn(activation)
#
#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos
#
#     def forward_post(self, src1, src2, hw_mean):
#         #src1：exemplar  src2：query
#         hw = self.hwsize(hw_mean)
#         b, _ = hw.shape
#         hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
#         src = self.multihead_attn1(query=hw,
#                                    key=src1,
#                                    value=src1)[0]
#         src = self.norm11(src)
#         src = self.linear12(self.dropout10(self.activation11(self.linear11(src))))
#         src = self.norm12(src)
#
#         srcs = self.multihead_attn2(query=src2,
#                                    key=src,
#                                    value=src)[0]
#         srcs = self.norm21(srcs)
#         srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
#         srcs = self.norm22(srcs)
#         return srcs.unsqueeze(0).transpose(1, 2)
#
#     def forward(self, src1, src2, hw_mean):
#         return self.forward_post(src1, src2, hw_mean)

class FeatureFusionLayerwyt_scaleNN2(nn.Module):

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

    def forward_post(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape
        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=hw,
                                   key=src1,
                                   value=src1)[0]
        src = hw + self.dropout11(src)
        src = self.norm11(src)
        src1 = self.linear12(self.dropout10(self.activation11(self.linear11(src))))
        src = src + self.dropout12(src1)
        src = self.norm12(src)

        srcs = self.multihead_attn2(query=src2,
                                   key=src,
                                   value=src)[0]
        srcs = self.norm21(srcs)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
        srcs = self.norm22(srcs)
        return srcs.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2, hw_mean):
        return self.forward_post(src1, src2, hw_mean)

class FeatureFusionLayerwyt(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2):
        src = self.multihead_attn(query=src2,
                                   key=src1,
                                   value=src1)[0]

        src = src + self.dropout1(src)
        src = self.norm1(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src)
        src = self.norm2(src)
        return src.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2):
        return self.forward_post(src1, src2)

class FeatureFusionLayerwyt_scale(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.hwsize = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512)
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

    def forward_post(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape
        # hw = hw.view(b, -1, 512).permute(1, 0, 2).contiguous()
        hw = hw.view(b, -1, 512).repeat(1, 64, 1).permute(1, 0, 2).contiguous()
        src = self.multihead_attn1(query=src1,
                                   key=hw,
                                   value=hw)[0]
        src = src + self.dropout11(src)
        src = self.norm11(src)
        src = self.linear12(self.dropout10(self.activation11(self.linear11(src))))
        src = src + self.dropout12(src)
        src = self.norm12(src)

        srcs = self.multihead_attn2(query=src2,
                                   key=src,
                                   value=src)[0]
        srcs = srcs + self.dropout21(srcs)
        srcs = self.norm21(srcs)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
        srcs = srcs + self.dropout22(srcs)
        srcs = self.norm22(srcs)
        return srcs.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2, hw_mean):
        return self.forward_post(src1, src2, hw_mean)

class FeatureFusionLayerwyt_scale1(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        #hw1
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

    def forward_post(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape
        # hw = hw.view(b, -1, 512).permute(1, 0, 2).contiguous()
        # hw = hw.view(b, -1, 512).repeat(1, 64, 1).permute(1, 0, 2).contiguous()
        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=src1,
                                   key=hw,
                                   value=hw)[0]
        src = src + self.dropout11(src)
        src = self.norm11(src)
        src = self.linear12(self.dropout10(self.activation11(self.linear11(src))))
        src = src + self.dropout12(src)
        src = self.norm12(src)

        srcs = self.multihead_attn2(query=src2,
                                   key=src,
                                   value=src)[0]
        srcs = srcs + self.dropout21(srcs)
        srcs = self.norm21(srcs)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
        srcs = srcs + self.dropout22(srcs)
        srcs = self.norm22(srcs)
        return srcs.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2, hw_mean):
        return self.forward_post(src1, src2, hw_mean)

class FeatureFusionLayerwyt_scale2(nn.Module):

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

    def forward_post(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape
        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=hw,
                                   key=src1,
                                   value=src1)[0]
        src = src + self.dropout11(src)
        src = self.norm11(src)
        src = self.linear12(self.dropout10(self.activation11(self.linear11(src))))
        src = src + self.dropout12(src)
        src = self.norm12(src)

        srcs = self.multihead_attn2(query=src2,
                                   key=src,
                                   value=src)[0]
        srcs = srcs + self.dropout21(srcs)
        srcs = self.norm21(srcs)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
        srcs = srcs + self.dropout22(srcs)
        srcs = self.norm22(srcs)
        return srcs.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2, hw_mean):
        return self.forward_post(src1, src2, hw_mean)

class FeatureFusionLayerwyt_scale3(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # hw default
        self.hwsize = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

        # FEM
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

    def forward_post(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape

        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=hw,
                                   key=src1,
                                   value=src1)[0]
        hw = hw + self.dropout11(src)
        hw = self.norm11(hw)
        src = self.linear12(self.dropout10(self.activation11(self.linear11(hw))))
        hw = hw + self.dropout12(src)
        hw = self.norm12(hw)

        srcs = self.multihead_attn2(query=src2,
                                   key=hw,
                                   value=hw)[0]
        src2 = src2 + self.dropout21(srcs)
        src2 = self.norm21(src2)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(src2))))
        src2 = src2 + self.dropout22(srcs)
        src2 = self.norm22(src2)
        return src2.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2, hw_mean):
        return self.forward_post(src1, src2, hw_mean)

class FeatureFusionLayerwyt_scale31(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # hw default
        self.hwsize = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

        # FEM
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

    def forward_post(self, src1, src2, hw_mean):
        #src1：exemplar  src2：query
        #true: sam    false: ffm
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape

        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=hw,
                                   key=src1,
                                   value=src1)[0]
        hw = hw + self.dropout11(src)
        hw = self.norm11(hw)
        src = self.linear12(self.dropout10(self.activation11(self.linear11(hw))))
        hw = hw + self.dropout12(src)
        hw = self.norm12(hw)

        srcs = self.multihead_attn2(query=src2,
                                    key=hw,
                                    value=hw)[0]
        srcs = srcs + self.dropout21(srcs)
        srcs = self.norm21(srcs)
        srcs = self.linear22(self.dropout20(self.activation21(self.linear21(srcs))))
        srcs = srcs + self.dropout22(srcs)
        srcs = self.norm22(srcs)
        return srcs.unsqueeze(0).transpose(1, 2)

    def forward(self, src1, src2, hw_mean):
        return self.forward_post(src1, src2, hw_mean)

class AblationFFN(nn.Module):

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # hw default
        self.hwsize = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

        # FEM
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.output_proj = nn.Conv2d(1, 512, kernel_size=1)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, tgt, hw_mean):
        hw = self.hwsize(hw_mean)
        b, _ = hw.shape

        hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=hw,
                                   key=memory,
                                   value=memory)[0]
        hw = hw + self.dropout11(src)
        hw = self.norm11(hw)

        tgt = tgt.transpose(0, 1)
        n1, hw1, c1 = tgt.shape
        h1 = w1 = int(math.sqrt(hw1))
        x = tgt.transpose(1, 2).reshape(n1, c1, h1, w1)

        hw = hw.transpose(0, 1)
        n2, hw2, c2 = hw.shape
        h2 = w2 = int(math.sqrt(hw2))
        y = hw.transpose(1, 2).reshape(n2, c2, h2, w2)
        # y = y.unsqueeze(1)
        tgt_list = []
        for x0, y0 in zip(x, y):
            tgt0 = F.conv2d(
                F.pad(x0.unsqueeze(0), ((int(w2 / 2)), int((w2 - 1) / 2), int(h2 / 2), int((h2 - 1) / 2))),
                y0.unsqueeze(0)
            )
            tgt_list.append(tgt0)
        tgt = torch.cat(tgt_list, dim=0)

        tgt = self.output_proj(tgt)
        tgt = tgt.reshape(n1, c1, h1*w1).transpose(1, 2)
        # tgt = tgt.transpose(0, 1)

        return tgt

class FeatureFusionLayerwyt_scale15_vis(nn.Module):

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

class Ablation_nohw(nn.Module):

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
        # hw = self.hwsize(hw_mean)
        # b, _ = hw.shape
        # hw = hw.view(b, 64, -1).repeat(1, 1, 512).permute(1, 0, 2).contiguous()  # shape2
        src = self.multihead_attn1(query=src1,
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

class FeatureFusionLayerwyt_scale16(nn.Module):

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
        src = self.multihead_attn1(query=src1,
                                   key=hw,
                                   value=hw)[0]
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

class FeatureFusionLayerwyt_scale15(nn.Module):

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
