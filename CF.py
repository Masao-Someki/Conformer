
import torch
import torch.nn as nn

from src import FFModule
from src import MHAModule
from src import KMeansMHA
from src import ConvModule
from src import Residual


class Conformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        ff1_hsize=1024,
        ff1_dropout=0,
        n_head=8,
        mha_dropout=0,
        kernel_size=3,
        conv_dropout=0,
        ff2_hsize=1024,
        ff2_dropout=0,
        batch_size=None,
        max_seq_length=512,
        window_size=128,
        decay=0.999,
        kmeans_dropout=0,
        is_left_to_right=False,
        is_share_qk=False,
        use_kmeans_mha=False,
    ):
        """Conformer Block.

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
            km_config (dict): Config for KMeans Attention.
            use_kmeans_mha(boolean): Flag to use KMeans Attention for multi-head attention.

        """
        super(Conformer, self).__init__()

        self.ff_module1 = Residual(
            module=FFModule(
                d_model=d_model,
                h_size=ff1_hsize,
                dropout=ff1_dropout
            ),
            half=True
        )
        if use_kmeans_mha:
            self.mha_module = Residual(
                module=KMeansMHA(
                    d_model=d_model,
                    n_head=n_head,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    window_size=window_size,
                    decay=decay,
                    dropout=kmeans_dropout,
                    is_left_to_right=is_left_to_right,
                    is_share_qk=is_share_qk,
                )
            )
        else:
            self.mha_module = Residual(
                module=MHAModule(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=mha_dropout
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
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, **kwargs):
        """Forward propagation of CF.

        Args:
            inputs (torch.Tensor): Input tensor. Shape is [B, L, D]

        Returns:
            torch.Tensor

        """
        x = self.ff_module1(inputs)
        x = self.mha_module(x, **kwargs)
        x = self.conv_module(x)
        x = self.ff_module2(x)
        x = self.layer_norm(x)
        return x

def get_conformer(config):
    return Conformer(
        d_model=config.d_model,
        ff1_hsize=config.ff1_hsize,
        ff1_dropout=config.ff1_dropout,
        n_head=config.n_head,
        mha_dropout=config.mha_dropout,
        kernel_size=config.kernel_size,
        conv_dropout=config.conv_dropout,
        ff2_hsize=config.ff2_hsize,
        ff2_dropout=config.ff2_dropout,
        km_config=config.km_config,
        use_kmeans_mha=config.use_kmeans_mha
    )
