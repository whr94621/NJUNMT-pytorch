import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import src.utils.init as my_init
from src.utils import nest

def sort_batch(seq_len):
    """Sorts torch tensor of integer indices by decreasing order."""
    with torch.no_grad():
        slens, sidxs = torch.sort(seq_len, descending=True)
    oidxs = torch.sort(sidxs)[1]

    return oidxs, sidxs, slens.tolist()

class RNN(nn.Module):

    def __init__(self, type, batch_first=False, **kwargs):

        super().__init__()

        self.type = type
        self.batch_first = batch_first

        # We always use batch first mode.
        if self.type == "gru":
            self.rnn = nn.GRU(batch_first=True, **kwargs)
        elif self.type == "lstm":
            self.rnn = nn.LSTM(batch_first=batch_first, **kwargs)

        self._reset_parameters()

    @property
    def batch_dim(self):
        if self.batch_first:
            return 0
        else:
            return 1

    def _reset_parameters(self):
        for weight in self.rnn.parameters():
            my_init.rnn_init(weight)

    def forward(self, input, input_mask, h_0=None):
        """
        :param input: Input sequence.
            With shape [batch_size, input_len, dim] if batch_first is True.

        :param input_mask: Mask of sequence.
        """

        self.rnn.flatten_parameters() # This is necessary if want to use DataParallel

        # Convert into batch first
        if self.batch_first is False:
            input = input.transpose(0,1).contiguous()
            input_mask = input_mask.transpose(0,1).contiguous()

        ##########################
        # Pad zero length with 1 #
        ##########################
        with torch.no_grad():
            seq_len = (1 - input_mask.long()).sum(1) # [batch_size, ]
            seq_len[seq_len.eq(0)] = 1

        out, h_n = self._forward_rnn(input, seq_len, h_0=h_0)

        if self.batch_first is False:
            out = out.transpose(0, 1).contiguous() # Convert to batch_second

        return out, h_n

    def _forward_rnn(self, input, input_length, h_0=None):
        """
        :param input: Input sequence.
            FloatTensor with shape [batch_size, input_len, dim]

        :param input_length: Mask of sequence.
            LongTensor with shape [batch_size, ]
        """
        total_length = input.size(1)

        # 1. Packed with pad
        oidx, sidx, slens = sort_batch(input_length)

        input_sorted = torch.index_select(input, index=sidx, dim=0)

        if h_0 is not None:
            h_0_sorted = nest.map_structure(lambda t: torch.index_select(t, 1, sidx), h_0)
        else:
            h_0_sorted = None

        # 2. RNN compute
        input_packed = pack_padded_sequence(input_sorted, slens, batch_first=True)

        out_packed, h_n_sorted = self.rnn(input_packed, h_0_sorted)

        # 3. Restore
        out_sorted = pad_packed_sequence(out_packed, batch_first=True, total_length=total_length)[0]
        out = torch.index_select(out_sorted, dim=0, index=oidx)

        h_n_sorted = nest.map_structure(lambda t: torch.index_select(t, 1, oidx), h_n_sorted)

        return out.contiguous(), h_n_sorted
