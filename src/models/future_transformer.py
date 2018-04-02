# Copyright 2017 Natural Language Processing Group, Nanjing University, zhengzx.142857@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Define a sequence-to-sequence model with attention. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from src.utils.common_utils import Vocab, GlobalNames
from src.modules.basic import BottleLinear as Linear
from src.modules.sublayers import LayerNorm, PositionwiseFeedForward, MultiHeadedAttention
from src.modules.embeddings import Embeddings
from src.utils.beam_search import tile_batch, tensor_gather_helper, mask_scores
from src.utils import nest

from src.models.transformer import get_attn_causal_mask
from src.models.transformer import Encoder, EncoderBlock
from src.models.transformer import Decoder, DecoderBlock
from src.models.transformer import Generator
from src.models.transformer import Transformer


def lower_triangle_matrix(tgt_len):
    """ Return a lower triangle 2-D matrix, whose lower parts are set to 1., otherwise 0.
    :return 2-D tensor like:
        1 0 0
        1 1 0
        1 1 1
        [tgt_len, tgt_len]
    """
    out = torch.tril(torch.ones(tgt_len, tgt_len))
    if GlobalNames.USE_GPU:
        out = out.cuda()
    return out


class TransformMinus(nn.Module):

    def __init__(self, dim_inp1, dim_inp2, dim_out, activ=nn.Tanh, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim_inp1, dim_out, bias=False)
        self.linear2 = nn.Linear(dim_inp2, dim_out, bias=False)
        self.linear_out = nn.Linear(dim_out, dim_out)
        self.activ = activ()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2):
        out = self.linear1(inp1) - self.linear2(inp2)
        out = self.activ(self.linear_out(out))

        return out


class LayerPostprocessing(nn.Module):
    """ Post Processing. Following: Dropout, Residual, LayerNorm"""

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, previous_x):
        return self.layer_norm(self.dropout(x) + previous_x)


class FutureBlock(nn.Module):
    """ Computing `Future` info and do post-processing"""

    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()

        self.history_linear = Linear(d_model, d_model, bias=True)
        self.transform_minus = TransformMinus(dim_inp1=d_model,
                                              dim_inp2=d_model,
                                              dim_out=d_model,
                                              activ=nn.Tanh,
                                              dropout=dropout)
        self.post_p = LayerPostprocessing(d_model, dropout=dropout)

    def forward(self, summarization, decoder_output):
        """
        :param summarization: Semantic summarization from Encoder.
            [batch, d_model]
        :param decoder_output: Decoder output from Decoder.
            [batch, tgt_len, d_model]
        :return: output. Future
            [batch, tgt_len, d_model]
        """
        batch_size, tgt_len, d_model = decoder_output.size()

        # prepare
        # tile summarization. [batch, tgt_len, d_model]
        summarization = torch.unsqueeze(summarization, 1).repeat(1, tgt_len, 1)
        # get lower triangle to obtain mask. [tgt_len, tgt_len]
        lower_triangle = lower_triangle_matrix(tgt_len)
        # tile to obtain history mask. [batch, tgt_len, tgt_len]
        history_mask = lower_triangle.unsqueeze(0).repeat(batch_size, 1, 1)

        # compute history. [batch, tgt_len, d_model]
        history_mask = Variable(history_mask.float())
        history = torch.bmm(history_mask, decoder_output) / history_mask.sum(-1, keepdim=True)

        history = torch.tanh(self.history_linear(history))

        # compute future. [batch, tgt_len, d_model]
        future = self.transform_minus(summarization, history)

        # post-processing. [batch, tgt_len, d_model]
        output = self.post_p(x=future, previous_x=decoder_output)

        return output


class InitialStateBridge(nn.Module):
    def __init__(self, d_model=512, activ=nn.Tanh, dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.activ = activ()
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_output, enc_mask):
        """
        :param enc_output: [batch, src_len, d_model]
        :param enc_mask: [batch, src_len]
        :return: out: [batch, d_model]
        """

        # prepare
        enc_mask = Variable(1 - enc_mask.float())

        # [batch, d_model]
        summarization = torch.bmm(enc_mask.unsqueeze(1), enc_output).squeeze(1)

        # [batch, 1]
        length = enc_mask.sum(1, keepdim=True)

        out = summarization / length

        out = self.activ(self.dropout(out))

        return out


class FutureTransformerDecoder(Decoder):
    """ A Transformer Decoder with modeling future information"""

    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8, d_word_vec=512,
            d_model=512, d_inner_hid=1024, dropout=0.1):
        super().__init__(n_tgt_vocab, n_layers, n_head, d_word_vec,
                         d_model, d_inner_hid, dropout)

        self.bridge = InitialStateBridge(d_model, dropout=dropout)
        self.future_block = FutureBlock(d_model, dropout=dropout)
        self.linear_combine_future = nn.Linear(d_model*2, d_model)
        self.pos_ffn_future = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_seq, enc_output, enc_mask, caches=None):
        batch_size, tgt_len = tgt_seq.size()
        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.data.eq(Vocab.PAD).unsqueeze(1).expand(batch_size, tgt_len, tgt_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, tgt_len, src_len)

        output = emb
        for i in range(self.num_layers):
            output, attn, all_input \
                = self.layer_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      cache=caches[i] if caches is not None else None)

        # compute future info.
        summarization = self.bridge(enc_output, enc_mask)

        future = self.future_block(summarization=summarization,
                                   decoder_output=output)

        # add future info. to output via pos_fnn
        combined = torch.cat([output, future], dim=-1)
        combined = self.linear_combine_future(combined)

        output = self.dropout(combined) + output
        output = self.pos_ffn_future(output)

        output = self.out_layer_norm(output)

        return output


class FutureTransformer(Transformer):
    """ A Transformer based sequence to sequence model with
        future information for Transformer decoder
    """

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024,
            dropout=0.1, proj_share_weight=True, **kwargs):
        super(FutureTransformer, self).__init__(
            n_src_vocab, n_tgt_vocab, n_layers, n_head,
            d_word_vec, d_model, d_inner_hid,
            dropout, proj_share_weight, **kwargs)
        # delete old Decoder
        delattr(self, "decoder")
        # add FutureTransformerDecoder
        self.decoder = FutureTransformerDecoder(
            n_tgt_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout)

        if "use_wpd" in kwargs:
            self.wpd_predictor = WPDPredictor(
                n_words=n_tgt_vocab,
                hidden_size=d_word_vec,
                shared_weight=self.decoder.embeddings.embeddings.weight,
                padding_idx=Vocab.PAD,
                **kwargs.get("wpd.params", None))

