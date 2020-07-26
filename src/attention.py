# This is an implementation of multi-head attention layer
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHAModule(nn.Module):
    def __init__(self, d_model, dropout=0.2, **kwargs):
        super(MHAModule, self).__init__()

        self.pe = PositionalEmbedding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha_RPE = MHAwithRelativePosEmb(d_model, dropout=dropout, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None, mems=None, head_mask=None):
        pos_emb = self.pe(inputs[0,:,0])
        x = self.layer_norm(inputs)
        x = self.mha_RPE(x, pos_emb, attn_mask, mems, head_mask)
        x = self.dropout(x)
        return x


class MHAwithRelativePosEmb(nn.Module):
    def __init__(self, d_model, n_head=1, dropout=0.2, epsilon=1e-5,
            r_r_bias=None, r_w_bias=None,):
        super(MHAwithRelativePosEmb, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0, 'd_model should be a multiple of n_head.'

        self.d_head = d_model // n_head

        self.qkv_net = nn.Linear(d_model, 3 * n_head * self.d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.o_net = nn.Linear(n_head * self.d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=epsilon)

        self.scale = 1 / (self.d_head ** 0.5)

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        # `r` is a positional embedding.
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            w_heads = self.qkv_net(self.layer_norm(cat))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(self.layer_norm(w))
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1  # Switch to bool
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum("ijbn,hbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)

        return attn_out


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        """Positional encoding module.
        This class is aquired from
        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_transfo_xl.py
        and modified a little by Masao Someki.
        """
        super(PositionalEmbedding, self).__init__()
        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]
