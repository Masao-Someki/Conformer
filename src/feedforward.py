# This is the pytorch implementation of feed forward layer

import torch
import torch.nn as nn

from .convs import swish

class FFModule(nn.Module):
    def __init__(self, d_model, h_size, dropout=0.2):
        super(FFModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer1 = nn.Linear(d_model, h_size)
        self.swish_activation = swish
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(h_size, d_model)

    def forward(self, inputs):
        x = self.layer_norm(inputs)
        x = self.layer1(x)
        x = self.swish_activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x
