#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Masao Someki

"""ConvolutionModule definition."""

import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """ConvolutionModule in Conformer model.
    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn
    """

    def __init__(self, in_channels, kernel_size=3, dropout=0.2, bias=False):
        """Construct an ConvolutionModule object."""
        super(ConvModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.layer_norm = nn.LayerNorm(in_channels)

        self.pos_conv1 = nn.Conv1d(
            in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.glu_activation = F.glu

        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            bias=bias
        )

        self.batch_norm = nn.BatchNorm1d(in_channels)

        self.swish_activation = swish

        self.pointwise_conv2 = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (Tensor): Shape of x is (Batch, Length, Dim)

        Returns:
            Tensor: (B, L, D)
        """
        x = self.layer_norm(x) # (B, L, D)
        x = self.pos_conv1(x.transpose(1, 2))  # (B, D*2, L)
        x = self.glu_activation(x, dim=1)  # (B, D, L)
        x = self.depthwise_conv(x) # (B, D, L)
        x = self.batch_norm(x) # (B, D, L)
        x = self.swish_activation(x) # (B, D, L)
        x = self.pointwise_conv2(x) # (B, D, L)
        x = self.dropout(x)

        return x.transpose(1, 2) # (B, L, D)
