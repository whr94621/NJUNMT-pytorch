import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import src.utils.nest as nest
from src.utils.common_utils import Vocab


def sort_batch(seq_mask, batch_dim=0):
    """Sorts torch tensor of integer indices by decreasing order."""

    omask = (1 - seq_mask).long()
    olens = omask.sum(1 - batch_dim)
    slens, sidxs = torch.sort(olens, descending=True)
    oidxs = torch.sort(sidxs)[1]

    return Variable(oidxs), Variable(sidxs), slens.tolist()

class RNN(nn.Module):

    def __init__(self, type, batch_first=False, **kwargs):

        super().__init__()

        self.type = type
        self.batch_first = batch_first

        if self.type == "gru":
            self.rnn = nn.GRU(batch_first=batch_first, **kwargs)
        elif self.type == "lstm":
            self.rnn = nn.LSTM(batch_first=batch_first, **kwargs)

    @property
    def batch_dim(self):
        if self.batch_first:
            return 0
        else:
            return 1

    def forward(self, input, input_mask, h_0=None):

        # 1. Packed with pad
        oidx, sidx, slens = sort_batch(input_mask, batch_dim=self.batch_dim)

        input_sorted = torch.index_select(input, dim=self.batch_dim, index=sidx)

        if h_0 is not None:

            if isinstance(h_0, tuple):
                h_0_sorted = (torch.index_select(h_0[0], dim=1, index=sidx),
                              torch.index_select(h_0[1], dim=1, index=sidx)
                              )
            else:
                h_0_sorted = torch.index_select(h_0, dim=1, index=sidx)
        else:
            h_0_sorted = None

        # 2. RNN compute
        input_packed = pack_padded_sequence(input=input_sorted, lengths=slens, batch_first=self.batch_first)

        out_packed, h_n_sorted = self.rnn(input_packed, h_0_sorted)

        # 3. Restore
        out_sorted = pad_packed_sequence(out_packed, batch_first=self.batch_first)[0]
        out = torch.index_select(out_sorted, dim=self.batch_dim, index=oidx)

        if isinstance(h_n_sorted, tuple):
            h_0_sorted = (torch.index_select(h_n_sorted[0], dim=1, index=oidx),
                          torch.index_select(h_n_sorted[1], dim=1, index=oidx)
                          )
        else:
            h_0_sorted = torch.index_select(h_n_sorted, dim=1, index=oidx)

        return out.contiguous(), h_0_sorted












