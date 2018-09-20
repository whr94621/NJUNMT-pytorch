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

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.init as my_init
from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.modules.cgru import CGRUCell
from src.modules.embeddings import Embeddings
from src.modules.rnn import RNN
from .base import NMTModel


class Encoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size
                 ):
        super(Encoder, self).__init__()

        # Use PAD
        self.embedding = Embeddings(num_embeddings=n_words,
                                    embedding_dim=input_size,
                                    dropout=0.0,
                                    add_position_embedding=False)

        self.gru = RNN(type="gru", batch_first=True, input_size=input_size, hidden_size=hidden_size,
                       bidirectional=True)

    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.detach().eq(PAD)

        emb = self.embedding(x)

        ctx, _ = self.gru(emb, x_mask)

        return ctx, x_mask


class Decoder(nn.Module):

    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 bridge_type="mlp",
                 dropout_rate=0.0):

        super(Decoder, self).__init__()

        self.bridge_type = bridge_type
        self.hidden_size = hidden_size
        self.context_size = hidden_size * 2

        self.embedding = Embeddings(num_embeddings=n_words,
                                    embedding_dim=input_size,
                                    dropout=0.0,
                                    add_position_embedding=False)

        self.cgru_cell = CGRUCell(input_size=input_size, hidden_size=hidden_size)

        self.linear_input = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.linear_ctx = nn.Linear(in_features=hidden_size * 2, out_features=input_size)

        self.dropout = nn.Dropout(dropout_rate)

        self._reset_parameters()

        self._build_bridge()

    def _reset_parameters(self):

        my_init.default_init(self.linear_input.weight)
        my_init.default_init(self.linear_hidden.weight)
        my_init.default_init(self.linear_ctx.weight)

    def _build_bridge(self):

        if self.bridge_type == "mlp":
            self.linear_bridge = nn.Linear(in_features=self.context_size, out_features=self.hidden_size)
            my_init.default_init(self.linear_bridge.weight)
        elif self.bridge_type == "zero":
            pass
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

    def init_decoder(self, context, mask):

        # Generate init hidden
        if self.bridge_type == "mlp":

            no_pad_mask = 1.0 - mask.float()
            ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)
            dec_init = F.tanh(self.linear_bridge(ctx_mean))

        elif self.bridge_type == "zero":
            batch_size = context.size(0)
            dec_init = context.new(batch_size, self.hidden_size).zero_()
        else:
            raise ValueError("Unknown bridge type {0}".format(self.bridge_type))

        dec_cache = self.cgru_cell.compute_cache(context)

        return dec_init, dec_cache

    def forward(self, y, context, context_mask, hidden, one_step=False, cache=None):

        emb = self.embedding(y)  # [seq_len, batch_size, dim]

        if one_step:
            (out, attn), hidden = self.cgru_cell(emb, hidden, context, context_mask, cache)
        else:
            # emb: [seq_len, batch_size, dim]
            out = []
            attn = []

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=1):
                (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(1), hidden, context, context_mask, cache)
                out += [out_t]
                attn += [attn_t]

            out = torch.stack(out).transpose(1, 0).contiguous()
            attn = torch.stack(attn).transpose(1, 0).contiguous()

        logits = self.linear_input(emb) + self.linear_hidden(out) + self.linear_ctx(attn)

        logits = F.tanh(logits)

        logits = self.dropout(logits)  # [seq_len, batch_size, dim]

        return logits, hidden


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):

        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.weight = shared_weight
        else:
            self._reset_parameters()

    def _reset_parameters(self):

        my_init.embedding_init(self.proj.weight)

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)


    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            return torch.nn.functional.softmax(logits, dim=-1)


class DL4MT(NMTModel):

    def __init__(self, n_src_vocab, n_tgt_vocab, d_word_vec, d_model, dropout,
                 proj_share_weight, bridge_type="mlp", **kwargs):

        super().__init__()

        self.encoder = Encoder(n_words=n_src_vocab, input_size=d_word_vec, hidden_size=d_model)

        self.decoder = Decoder(n_words=n_tgt_vocab, input_size=d_word_vec, hidden_size=d_model,
                               dropout_rate=dropout, bridge_type=bridge_type)

        if proj_share_weight is False:
            generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD)
        else:
            generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD,
                                  shared_weight=self.decoder.embedding.embeddings.weight
                                  )
        self.generator = generator

    def forward(self, src_seq, tgt_seq, log_probs=True):

        ctx, ctx_mask = self.encoder(src_seq)

        dec_init, dec_cache = self.decoder.init_decoder(ctx, ctx_mask)

        logits, _ = self.decoder(tgt_seq,
                                 context=ctx,
                                 context_mask=ctx_mask,
                                 one_step=False,
                                 hidden=dec_init,
                                 cache=dec_cache)  # [tgt_len, batch_size, dim]

        return self.generator(logits, log_probs)

    def encode(self, src_seq):

        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        dec_init, dec_caches = self.decoder.init_decoder(context=ctx, mask=ctx_mask)

        if expand_size > 1:
            ctx = tile_batch(ctx, expand_size)
            ctx_mask = tile_batch(ctx_mask, expand_size)
            dec_init = tile_batch(dec_init, expand_size)
            dec_caches = tile_batch(dec_caches, expand_size)

        return {"dec_hiddens": dec_init, "dec_caches": dec_caches, "ctx": ctx, "ctx_mask": ctx_mask}

    def decode(self, tgt_seq, dec_states, log_probs=True):

        ctx = dec_states['ctx']
        ctx_mask = dec_states['ctx_mask']

        dec_hiddens = dec_states['dec_hiddens']
        dec_caches = dec_states['dec_caches']

        final_word_indices = tgt_seq[:, -1].contiguous()

        logits, next_hiddens = self.decoder(final_word_indices, hidden=dec_hiddens, context=ctx, context_mask=ctx_mask,
                                            one_step=True, cache=dec_caches)

        scores = self.generator(logits, log_probs=log_probs)

        dec_states = {"ctx": ctx, "ctx_mask": ctx_mask, "dec_hiddens": next_hiddens, "dec_caches": dec_caches}

        return scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        dec_hiddens = dec_states["dec_hiddens"]

        batch_size = dec_hiddens.size(0) // beam_size

        dec_hiddens = tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=dec_hiddens,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, -1])

        dec_states['dec_hiddens'] = dec_hiddens

        return dec_states
