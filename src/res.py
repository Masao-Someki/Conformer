# pytorch implementation of residual normalization layer

import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module, half=False):
        super(Residual, self).__init__()
        self.net = module
        self.half = half

    def forward(self, inputs, **kwargs):
        x = self.net(inputs, **kwargs)

        if self.half:
            return (x * 0.5) + inputs
        else:
            return x + inputs
