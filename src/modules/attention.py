import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import src.utils.init as my_init

from .basic import BottleSoftmax



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=1)

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
            attn = attn.masked_fill(Variable(attn_mask), -1e18)

        attn = self.softmax(attn)
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

        self.softmax = nn.Softmax(dim=0)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.parameters():
            my_init.default_init(weight.data)

    def compute_cache(self, memory):

        return self.linear_key(memory)


    def forward(self, query, memory, cache=None, mask=None):
        """
        :param query: Key tensor.
            with shape [batch_size, input_size]

        :param memory: Memory tensor.
            with shape [men_len, batch_size, input_size]

        :param mask: Memory mask which the PAD position is marked with true.
            with shape [mem_len, batch_size]
        """
        q = self.linear_query(query)

        if cache is not None:
            k = cache
        else:
            k = self.linear_key(memory)

        logit = q.unsqueeze(0) + k # [mem_len, batch_size, dim]
        logit = F.tanh(logit)
        logit = self.linear_logit(logit).squeeze(2) # [mem_len, batch_size, 1] ==> [mem_len, batch_size]

        if mask is not None:
            logit = logit.masked_fill(Variable(mask), -1e18)

        weights = self.softmax(logit) # [mem_len, batch_size]

        attns = (weights.unsqueeze(2) * memory).sum(0)

        return attns, weights