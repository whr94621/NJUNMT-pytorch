import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils.init as my_init

from .tensor_utils import tile_batch, FLOAT32_INF


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        :type attn_mask: torch.FloatTensor
        :param attn_mask: Mask of the attention.
            3D tensor with shape [batch_size, time_step_key, time_step_value]
        """
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())
            attn = attn.masked_fill(attn_mask, -FLOAT32_INF)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class BahdanauAttention(nn.Module):

    def __init__(self, query_size, key_size, hidden_size=None):
        super().__init__()

        self.query_size = query_size
        self.key_size = key_size

        if hidden_size is None:
            hidden_size = key_size

        self.hidden_size = hidden_size

        self.linear_key = nn.Linear(in_features=self.key_size, out_features=self.hidden_size)
        self.linear_query = nn.Linear(in_features=self.query_size, out_features=self.hidden_size)
        self.linear_logit = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.tanh = nn.Tanh()

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.parameters():
            my_init.default_init(weight)

    def compute_cache(self, memory):

        return self.linear_key(memory)

    def forward(self, query, memory, cache=None, mask=None):
        """
        :param query: Key tensor.
            with shape [batch_size, input_size]

        :param memory: Memory tensor.
            with shape [batch_size, mem_len, input_size]

        :param mask: Memory mask which the PAD position is marked with true.
            with shape [batch_size, mem_len]
        """

        if query.dim() == 2:
            query = query.unsqueeze(1)
            one_step = True
        else:
            one_step = False

        batch_size, q_len, q_size = query.size()
        _, m_len, m_size = memory.size()

        q = self.linear_query(query.view(-1, q_size))  # [batch_size, q_len, hidden_size]

        if cache is not None:
            k = cache
        else:
            k = self.linear_key(memory.view(-1, m_size))  # [batch_size, m_len, hidden_size]

        # logit = q.unsqueeze(0) + k # [mem_len, batch_size, dim]
        logits = q.view(batch_size, q_len, 1, -1) + k.view(batch_size, 1, m_len, -1)
        logits = self.tanh(logits)
        logits = self.linear_logit(logits.view(-1, self.hidden_size)).view(batch_size, q_len, m_len)

        if mask is not None:
            mask_ = mask.unsqueeze(1)  # [batch_size, 1, m_len]
            logits = logits.masked_fill(mask_, -1e18)

        weights = F.softmax(logits, dim=-1)  # [batch_size, q_len, m_len]

        # [batch_size, q_len, m_len] @ [batch_size, m_len, m_size]
        # ==> [batch_size, q_len, m_size]
        attns = torch.bmm(weights, memory)

        if one_step:
            attns = attns.squeeze(1)  # ==> [batch_size, q_len]

        return attns, weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1, exclude_diagonal=False):

        super(MultiHeadedAttention, self).__init__()

        if dim_per_head is None:
            assert model_dim % head_count == 0
            dim_per_head = model_dim // head_count

        self.head_count = head_count

        self.dim_per_head = dim_per_head

        self.model_dim = model_dim

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.dim_per_head * head_count, model_dim)
        self.exclude_diagonal = exclude_diagonal

    def _split_heads(self, x):

        batch_size = x.size(0)

        # [batch_size * n_head, seq_len, dim_per_head]
        return x.view(batch_size, -1, self.head_count, self.dim_per_head) \
            .transpose(1, 2).contiguous().view(batch_size * self.head_count, -1, self.dim_per_head)

    def _combine_heads(self, x):

        """:param x: [batch_size * head_count, seq_len, dim_per_head]"""
        seq_len = x.size(1)

        return x.view(-1, self.head_count, seq_len, self.dim_per_head).transpose(1, 2).contiguous() \
            .view(-1, seq_len, self.head_count * self.dim_per_head)

    def forward(self, key, value, query, mask=None, enc_attn_cache=None, self_attn_cache=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key))  # [batch_size * num_head, seq_len, dim_head]
            value_up = self._split_heads(self.linear_values(value))

        if self_attn_cache is not None:
            key_up_prev, value_up_prev = self_attn_cache
            # Append current key and value to the cache
            key_up = torch.cat([key_up_prev, key_up], dim=1)
            value_up = torch.cat([value_up_prev, value_up], dim=1)

        query_up = self._split_heads(self.linear_query(query))

        key_len = key_up.size(1)
        query_len = query_up.size(1)

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.bmm(query_up, key_up.transpose(1, 2))

        if mask is not None:
            mask = tile_batch(mask, self.head_count)
            scores = scores.masked_fill(mask, -1e18)

        if self.exclude_diagonal:
            mask = torch.arange(0, scores.size(2)).to(scores).long()
            mask = mask.expand(scores.size(0), 1, mask.size(-1))
            scores.scatter_(1, mask, -FLOAT32_INF)

        # 3) Apply attention dropout and compute context vectors.
        attn = F.softmax(scores, dim=-1)  # [bsz * n_head, q_len, k_len]
        drop_attn = self.dropout(attn)
        context = self._combine_heads(torch.bmm(drop_attn, value_up))

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()
        # END CHECK
        return output, top_attn, [key_up, value_up]
