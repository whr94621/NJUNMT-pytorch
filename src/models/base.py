# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch.nn as nn


class NMTModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, src_seq, tgt_seq, **kwargs):
        """ Forward process of NMT model

        Given source and target side, return the probabilities of the samples.
        """
        raise NotImplementedError

    def init_decoder(self, enc_outputs, expand_size=1):
        """
        Prepare for decoding
        Args:
            enc_outputs (dict): Output dictionary from the return value of ```encode```
            expand_size: (int): Repeat for several times along the first dimension. This is usefull for
                beam search

        Returns (dict):
            A dict object store the states in decoding phase.
        """
        raise NotImplementedError

    def encode(self, src_seq):
        """
        Encode the source side
        """
        raise NotImplementedError

    def decode(self, tgt_seq, dec_states, log_probs=True):
        """
        Decoding for one step
        Args:
            tgt_seq (torch.Tensor): All the generated tokens before.
            dec_states (dict): Decoding states.
            log_probs (bool): Return logarithm probabilities or probabilities. Default is True.

        Returns:
            Scores of next tokens and decoding states.
        """

        raise NotImplementedError

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        """
        Re-ordering decoding states according to newly selected beam indices

        Args:
            dec_states (dict):
            new_beam_indices (torch.Tensor):
            beam_size (int):

        Returns:
            Re-ordered dec_states
        """
        raise NotImplementedError
