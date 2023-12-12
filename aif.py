#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from mask import make_pad_mask, create_aif_mask
from attention import DotProductAttention


class AIF(torch.nn.Module):
    """Auto-regressive Integrate-and-Fire (AIF).

    LABEL-SYNCHRONOUS NEURAL TRANSDUCER FOR END-TO-END ASR
    https://arxiv.org/pdf/2307.03088.pdf
    """

    def __init__(
        self,
        encoder_dim,
        predictor_dim
    ):
        super().__init__()

        self.proj = torch.nn.Linear(encoder_dim-1, predictor_dim)
        self.att = DotProductAttention()

    def forward(
        self,
        xs,
        xs_lens,
        ys,
        ys_lens,
    ):
        """
        Args:
            xs (torch.Tensor): Acoustic encoder output, size
                (B, T, D').
            ys (torch.Tensor): Prediction network intermediate output, size
                (B, V, D).
            ys_lens (torch.Tensor): Target lengths, size (B,)
            xs_lens (torch.Tensor): Encoder ouput lengths, size (B,)

        Returns:
            torch.Tensor: Transformed value (B, V, D)
        """
        # B,T,1
        # employs a simple method to find the weight αt by applying a sigmoid 
        # function to the last element et,d of each encoder output frame 
        alphas = torch.sigmoid(xs[:,:,-1].unsqueeze(-1))

        encoder_mask = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        alphas = alphas * encoder_mask.transpose(-1, -2).type_as(alphas)

        # B,T,1 -> B,T
        alphas = alphas.squeeze(-1)

        # B,T -> B
        token_num = alphas.sum(-1)

        mask, csum = create_aif_mask(alphas, ys_lens, encoder_mask)

        # mapped to the same dimension as ys
        xs_proj = self.proj(xs[:,:,:-1])

        acoustic_embeds = self.att(
            ys,
            xs_proj,
            xs_proj,
            mask=mask.to(ys.device)
        )

        return acoustic_embeds, token_num, alphas, csum
