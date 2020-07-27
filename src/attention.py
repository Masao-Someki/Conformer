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
        self.mha_RPE = RelPositionMultiHeadedAttention(d_model, dropout=dropout, **kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask=None, mems=None, head_mask=None):
        pos_emb = self.pe(inputs[0,:,0])
        x = self.layer_norm(inputs)
        x = self.mha_RPE(x, pos_emb, mems, attn_mask)
        x = self.dropout(x)
        return x


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    This class is aquired from 
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/attention.py
    (Apache2.0 Licence) and modified a little by Masao-Someki

    Paper: https://arxiv.org/abs/1901.02860
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, d_model, n_head=4, dropout=0.2):
        """Construct an RelPositionMultiHeadedAttention object."""
        super(RelPositionMultiHeadedAttention, self).__init__()
        # linear transformation for positional ecoding
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.qkv_net = nn.Linear(d_model, d_model * 3, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        :param torch.Tensor x: (batch, time, size)
        :param bool zero_triu: return the lower triangular part of the matrix
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward_qkv(self, w, mem):
        """Transform query, key and value.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :return torch.Tensor transformed query, key and value
        """
        bsz = w.size(0)
        if mem is not None:
            cat = torch.cat([mem, w], 0)
            w_heads = self.qkv_net(self.layer_norm(cat))
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(self.layer_norm(w))
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        qlen = w.size(1)
        klen = w_head_k.size(1)
        w_head_q = w_head_q.view(bsz, qlen, self.n_head, self.d_k)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(bsz, klen, self.n_head, self.d_k)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(bsz, klen, self.n_head, self.d_k)  # qlen x bsz x n_head x d_head

        return w_head_q, w_head_k.transpose(1, 2), w_head_v.transpose(1, 2)

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor scores: (batch, time1, time2)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor transformed `value` (batch, time2, d_model)
            weighted by the attention score (batch, time1, time2)
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_head * self.d_k)
        )  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, pos_emb, mem=None, mask=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor pos_emb: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q, k, v = self.forward_qkv(query, mem)
        # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)



class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        """Positional encoding module.
        This class is aquired from
        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_transfo_xl.py
        (Apache2.0 Licence) and modified a little by Masao Someki.
        """
        super(PositionalEmbedding, self).__init__()
        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1).transpose(0, 1)
        else:
            return pos_emb[:, None, :].transpose(0, 1)
