# This is an implementation of multi-head attention layer
import math
import sys

import torch
import torch.nn as nn


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

class KMeansMHA(nn.Module):
    """KMeans Atteniton module.
    This implementation is based on the following paper:
    
    Efficient Content-Based Sparse Attention with Routing Transformers
    https://arxiv.org/abs/2003.05997

    Args:
        d_mdoel (int): dimension
        n_head (int) : Number of heads
        batch_size (int): Batch size.
        max_seq_length (int):  Max length of input
        window_size(int) : window size.
        decay (float) : decay.
        dropout (float): dropout.
        is_left_to_right(boolean) : Flag to apply tril mask in attention.
        is_share_qk (boolean) : flag to share Q and K in attention.

    """
    def __init__(self,
        d_model=512,
        n_head=8,
        batch_size=32,
        max_seq_length=512,
        window_size=128,
        decay=0.999,
        dropout=0,
        is_left_to_right=False,
        is_share_qk=False,
    ):
        super(KMeansMHA, self).__init__()

        self.heads = n_head
        self.kmeans_att = KMA(
            batch_size=batch_size,
            n_head=n_head,
            d_model=d_model,
            max_seq_length=max_seq_length,
            window_size=window_size,
            decay=decay,
            is_left_to_right=is_left_to_right
        )
        self.is_share_qk = is_share_qk

        # attention settings
        if (is_share_qk):
            self.to_q = nn.Linear(d_model, d_model)
            self.to_v = nn.Linear(d_model, d_model)
        else:
            self.to_q = nn.Linear(d_model, d_model)
            self.to_k = nn.Linear(d_model, d_model)
            self.to_v = nn.Linear(d_model, d_model)

    def forward(self, inputs, context=None):
        """Forward propagation.

        Args:
            inputs: (B, L, D)
            context: (B, L, D)

        Returns:
            Tensor: (B, L, D)
        """
        q = self.to_q(inputs)
        v = self.to_v(inputs) if context is None else self.to_v(context)

        # if left to right mask then:
        if self.is_share_qk:
            k = q
        else:
            k = self.to_k(inputs) if context is None else self.to_k(context)

        Q = self.split_heads(q)
        K = self.split_heads(k)
        V = self.split_heads(v)

        x = self.kmeans_att(Q, K, V)
        x = self.merge_head(x)
        return x

    def split_heads(self, x):
        """Split head and reshape

        Args:
            x: (B, L, D)

        Returns:
            Tensor: (B, H, L, D)
        """
        b, l, d = x.shape
        return x.reshape((b, l, self.heads, d // self.heads)).transpose(1, 2)

    def merge_head(self, x):
        """Merge head

        Args:
            x: (B, H, L, D)

        Returns:
            Tensor: (B, L, D)
        """
        b, h, l, d = x.shape
        x = x.transpose(1, 2)
        return x.reshape((b, l, -1))


class KMA(nn.Module):
    """
    Computes KMeans Attention.
    I refered the following repository a lot
    to implement gather/scatter functions.
    https://github.com/lucidrains/routing-transformer
        
    """
    def __init__(self,
        batch_size=32,
        n_head=8,
        d_model=512,
        max_seq_length=1024,
        window_size=128,
        decay=0.999,
        is_left_to_right=False
    ):
        super(KMA, self).__init__()
        self.layernorm = nn.LayerNorm(
            (max_seq_length, d_model // n_head),
            elementwise_affine=False
        )
        self.is_left_to_right = is_left_to_right
        self.softmax = nn.Softmax(dim=-1)
        q_scatter_base = torch.zeros((batch_size, n_head, max_seq_length, d_model // n_head))
        self.register_buffer('q_scatter_base', q_scatter_base)
        o_scatter_base = torch.zeros((batch_size, n_head, max_seq_length, d_model // n_head))
        self.register_buffer('o_scatter_base', o_scatter_base)

        # k-means settings
        # you can add initialize mu parameter.
        # In the paper, window size is defined as:
        # w <- n/k
        # so that the number of the clusters k is:
        # k = n/w
        self.k = max_seq_length // window_size
        self.w = n_head // self.k
        mu = torch.rand((self.k, d_model // n_head), requires_grad=False)
        self.register_buffer('mu', mu)
        self.decay = decay

        # attention mask.
        if is_left_to_right:
            mask = torch.tril(
                torch.ones((max_seq_length, max_seq_length), requires_grad=False)
                )
            mask[self.mask == 0] = float('-inf')
            self.register_buffer('mask', mask)

    def forward(self, Q, K, V):
        """Forward propagation.

        This implementation is based on the Algorithm 1
        in the Routing Transformer paper.

        Args:
            inputs: (batch, head, length, dim)
            context: (batch, head, length, dim)

        Returns:
            Tensor: (batch, head, length, dim)
        """
        b, h, l, d = Q.shape
        with torch.no_grad():
            # Normalize to unit ball
            Q = self.layernorm(Q) # scale, bias disabled
            K = self.layernorm(K) # scale, bias disabled

            Q_prod = torch.matmul(self.mu, Q.transpose(2, 3)) # dot production, k * n
            if not self.is_left_to_right:
                K_prod = torch.matmul(self.mu, K.transpose(2, 3)) # dot production

            _, Q_idx = torch.topk(Q_prod, self.w)
            Q_idx, _ = torch.sort(Q_idx)
            Q_idx = Q_idx.reshape(*Q_idx.size()[:2], -1)
            K_idx = Q_idx

            if not self.is_left_to_right:
                _, K_idx = torch.topk(K_prod, self.w)
                K_idx, _ = torch.sort(K_idx)
                K_idx = K_idx.reshape(*K_idx.size()[:2], -1)

            # update centroids
            Q_m = torch.nn.functional.one_hot(
                torch.argmax(Q_prod, dim=-1),
                num_classes=Q_prod.shape[-1]
            ).float()
            K_m = torch.nn.functional.one_hot(
                torch.argmax(K_prod, dim=-1),
                num_classes=K_prod.shape[-1]
            ).float()
            self.mu = self.decay * self.mu \
                + (1 - self.decay) * torch.matmul(Q_m, Q/2) \
                + (1 - self.decay) * torch.matmul(K_m, K/2)

        # X_dash : (B, nh, k*w, d)
        Q_dash = torch.gather(Q, 2, expand_dim(Q_idx, -1, d)).reshape(b, h, self.k, self.w, d)
        K_dash = torch.gather(K, 2, expand_dim(K_idx, -1, d)).reshape(b, h, self.k, self.w, d)
        V_dash = torch.gather(V, 2, expand_dim(K_idx, -1, d)).reshape(b, h, self.k, self.w, d)
        
        A = torch.matmul(Q_dash, K_dash.transpose(3, 4))
        if self.is_left_to_right:
            A = self.mask + A

        A = self.softmax(A)
        V_dash = torch.matmul(A, V_dash).reshape(b, h, -1, d) # (b, h, k * w, d)
        X = self.scatter(2, K_idx.unsqueeze(-1).expand_as(V_dash), V_dash)

        return X

    def scatter(self, dim, idx, v):
        numer = self.q_scatter_base.scatter_add(dim, idx, v)
        denom = self.q_scatter_base.scatter_add(dim, idx, self.o_scatter_base)
        return numer / (denom + 1e-5)

        
