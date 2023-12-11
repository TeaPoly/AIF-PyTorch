#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch


class DotProductAttention(torch.nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values, mask):
        # (B, V, D) * (B, D, T) -> (B, V, T)
        scores = torch.bmm(
            queries, keys.transpose(-2, -1))

        if mask is not None:
            mask = mask.eq(0)  # (B, V, T)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (B, V, T)
        else:
            attn = torch.softmax(scores, dim=-1)

        # (B, V, T) * (B, T, D) -> (B, V, D)
        return torch.bmm(attn, values)
