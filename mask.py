#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Optional

import torch


def make_pad_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = int(lengths.size(0))
    if max_len is None:
        max_len = int(lengths.max().item())
    seq_range = torch.arange(
        0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def create_aif_mask(
    alphas: torch.Tensor,
    ys_lens: torch.Tensor,
    encoder_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Auto-regressive Integrate-and-Fire (AIF).

    LABEL-SYNCHRONOUS NEURAL TRANSDUCER FOR END-TO-END ASR
    https://arxiv.org/pdf/2307.03088.pdf

    alphas: tensor([[0.1576, 0.1511, 0.5337, 0.1128, 0.1766, 0.5147, 0.0855, 0.4968, 0.5363,
             0.4238],
            [0.3931, 0.4622, 0.5855, 0.3605, 0.0933, 0.6409, 0.5778, 0.3256, 0.1385,
             0.2042]])

    mask: tensor([[[ True,  True,  True,  True,  False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  False, False, False],
             [ True,  True,  True,  True,  True,  True,  True,  True,  True,  False]],

            [[ True,  True,  False, False, False, False, False, False, False, False],
             [ True,  True,  True,  True,  True,  False, False, False, False, False],
             [ True,  True,  True,  True,  True,  True,  False, False, False, False]]])

    Args:
        alphas (torch.Tensor): Mask, size (B, T).
        ys_lens (torch.Tensor): Target lengths, size (B,)
        encoder_mask (torch.Tensor): Mask, size (B, 1, T)

    Returns:
        torch.Tensor: Transformed value (B, V, D)

    """
    B, T = alphas.size()
    V = ys_lens.max()
    csum = alphas.cumsum(-1)
    v_range = torch.arange(V).view(1, -1, 1).to(alphas.device)
    mask = csum.unsqueeze(1) <= v_range+1

    m = ~make_pad_mask(ys_lens).unsqueeze(-1)  # (B, V, 1)
    mask = mask & m

    if encoder_mask is not None:
        mask = mask & encoder_mask

    return mask, csum
