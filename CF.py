
import torch
import torch.nn as nn

from src import FFModule
from src import MHAModule
from src import ConvModule
from src import Residual


class Conformer(nn.Module):
    def __init__(
        self,
        d_model,
        ff1_hsize=1024,
        ff1_dropout=0.2,
        n_head=4,
        mha_dropout=0.2,
        epsilon=1e-5,
        kernel_size=3,
        conv_dropout=0.2,
        ff2_hsize=1024,
        ff2_dropout=0.2
    ):
        """RNN enhanced Transformer Block.

        Args:
            d_model (int): Embedded dimension of input.
            ff1_hsize (int): Hidden size of th first FFN
            ff1_drop (float): Dropout rate for the first FFN
            n_head (int): Number of heads for MHA
            mha_dropout (float): Dropout rate for the first MHA
            epsilon (float): Epsilon
            kernel_size (int): Kernel_size for the Conv
            conv_dropout (float): Dropout rate for the first Conv
            ff2_hsize (int): Hidden size of th first FFN
            ff2_drop (float): Dropout rate for the first FFN

        """
        super(Conformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ff_module1 = Residual(
            module=FFModule(
                d_model=d_model,
                h_size=ff1_hsize,
                dropout=ff1_dropout
            ),
            half=True
        )
        self.mha_module = Residual(
            module=MHAModule(
                d_model=d_model,
                n_head=n_head,
                dropout=mha_dropout,
                epsilon=epsilon
            )
        )
        self.conv_module = Residual(
            module=ConvModule(
                in_channels=d_model,
                kernel_size=kernel_size,
                dropout=conv_dropout
            )
        )
        self.ff_module2 = Residual(
            FFModule(
                d_model=d_model,
                h_size=ff2_hsize,
                dropout=ff2_dropout
            ),
            half=True
        )

    def forward(self, inputs, **kwargs):
        """Forward propagation of RT.

        Args:
            inputs (torch.Tensor): Input tensor. Shape is [B, L, D]
            mask (torch.Tensor): Mask for attention. If None, calculation
                of masking will not computed.

        Returns:
            torch.Tensor

        """
        x = self.ff_module1(inputs)
        x = self.mha_module(x, **kwargs)
        x = self.conv_module(x)
        x = self.ff_module2(x)

        return x

def get_conformer(d_model, **kwargs):
    return Conformer(d_model, **kwargs)
