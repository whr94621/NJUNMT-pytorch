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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.tensor_utils import tile_batch
from src.decoding.utils import tensor_gather_helper
from src.models.base import NMTModel
from src.modules.activation import GELU
from src.modules.attention import MultiHeadedAttention
from src.modules.basic import BottleLinear as Linear
from src.modules.embeddings import Embeddings
from src.utils import nest
from src.utils.common_utils import Constants

PAD = Constants.PAD

__all__ = [
    'Transformer',
    'transformer_base_v1',
    'transformer_base_v2',
    'transformer_low_resource'
]


def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class PositionwiseFeedForward(nn.Module):
    """
    A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        size (int): the size of input for the first-layer of the FFN.
        hidden_size (int): the hidden layer size of the second-layer
                          of the FNN.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        # Save a little memory, by doing inplace.
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "gelu":
            self.activation = GELU()
        else:
            raise ValueError

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Block(nn.Module):
    """
    The definition of block (sublayer) is formally introduced in Chen, Mia Xu et al.
    “The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation.” ACL (2018).

    A block is consist of a transform function (TF), a layer normalization (LN) and a residual connection with
    dropout (Drop-Res). There are two kinds of block, differing in the position of layer normalization:
        a): LN -> TF -> Drop-Res  (layer_norm_first is True)
        b): TF -> Drop-Res -> LN

    A block can return more than one output, but we only perform LN and Drop-Res on the first output.
    """

    def __init__(self, size, dropout, layer_norm_first=True):
        super().__init__()

        self.layer_norm_first = layer_norm_first

        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def _transform(self, *args, **kwargs):

        raise NotImplementedError

    def forward(self, x, *args, **kwargs):

        # 1. layer normalization
        if self.layer_norm_first:
            transform_input = self.layer_norm(x)
        else:
            transform_input = x

        # 2. transformation
        output = self._transform(transform_input, *args, **kwargs)

        # 3. dropout & residual add
        if not isinstance(output, tuple):
            output = x + self.dropout(output)
            if not self.layer_norm_first:
                output = self.layer_norm(output)
        else:
            output = (x + self.dropout(output[0]),) + output[1:]
            if not self.layer_norm_first:
                output = (self.layer_norm(output[0]),) + output[1:]

        return output


class SelfAttentionBlock(Block):

    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1, attn_dropout=0.1, layer_norm_firs=True):
        super().__init__(model_dim, dropout=dropout, layer_norm_first=layer_norm_firs)

        self.transform_layer = MultiHeadedAttention(model_dim=model_dim, head_count=head_count,
                                                    dim_per_head=dim_per_head, dropout=attn_dropout)

    def _transform(self, x, mask=None, self_attn_cache=None):
        return self.transform_layer(x, x, x, mask=mask, self_attn_cache=self_attn_cache)


class EncoderAttentionBlock(Block):

    def __init__(self, model_dim, head_count, dim_per_head=None, dropout=0.1, attn_dropout=0.1, layer_norm_first=True):
        super().__init__(model_dim, dropout=dropout, layer_norm_first=layer_norm_first)

        self.transform_layer = MultiHeadedAttention(model_dim=model_dim, head_count=head_count,
                                                    dim_per_head=dim_per_head, dropout=attn_dropout)

    def _transform(self, dec_hidden, context, mask=None, enc_attn_cache=None):
        return self.transform_layer(context, context, dec_hidden, mask=mask, enc_attn_cache=enc_attn_cache)


class PositionwiseFeedForwardBlock(Block):

    def __init__(self, size, hidden_size, dropout=0.1, layer_norm_first=True, activation="relu"):
        super().__init__(size=size, dropout=dropout, layer_norm_first=layer_norm_first)

        self.transform_layer = PositionwiseFeedForward(size=size, hidden_size=hidden_size, dropout=dropout,
                                                       activation=activation)

    def _transform(self, x):
        return self.transform_layer(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1, layer_norm_first=True,
                 ffn_activation="relu"):
        super(EncoderLayer, self).__init__()

        self.slf_attn = SelfAttentionBlock(head_count=n_head, model_dim=d_model, dropout=dropout,
                                           dim_per_head=dim_per_head, layer_norm_firs=layer_norm_first)

        self.pos_ffn = PositionwiseFeedForwardBlock(size=d_model, hidden_size=d_inner_hid, dropout=dropout,
                                                    layer_norm_first=layer_norm_first, activation=ffn_activation)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        context, _, _ = self.slf_attn(enc_input, mask=slf_attn_mask)

        return self.pos_ffn(context)


class Encoder(nn.Module):

    def __init__(
            self, n_src_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None,
            padding_idx=PAD, positional_embedding="sin", layer_norm_first=True, ffn_activation="relu"):
        super().__init__()

        self.scale = d_word_vec ** 0.5
        self.num_layers = n_layers
        self.layer_norm_first = layer_norm_first

        self.embeddings = Embeddings(num_embeddings=n_src_vocab,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     positional_embedding=positional_embedding
                                     )

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                          dim_per_head=dim_per_head, layer_norm_first=layer_norm_first, ffn_activation=ffn_activation)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        if not self.layer_norm_first:
            emb = self.layer_norm(emb)

        out = emb

        for i in range(self.num_layers):
            out = self.layer_stack[i](out, enc_slf_attn_mask)

        if self.layer_norm_first:
            out = self.layer_norm(out)

        return out, enc_mask


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1, layer_norm_first=True,
                 ffn_activation="relu"):
        super(DecoderLayer, self).__init__()

        self.slf_attn = SelfAttentionBlock(head_count=n_head, model_dim=d_model, dropout=dropout,
                                           dim_per_head=dim_per_head, layer_norm_firs=layer_norm_first)
        self.ctx_attn = EncoderAttentionBlock(head_count=n_head, model_dim=d_model, dropout=dropout,
                                              dim_per_head=dim_per_head, layer_norm_first=layer_norm_first)

        self.pos_ffn = PositionwiseFeedForwardBlock(size=d_model, hidden_size=d_inner_hid,
                                                    layer_norm_first=layer_norm_first, activation=ffn_activation)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        query, _, self_attn_cache = self.slf_attn(dec_input, mask=slf_attn_mask, self_attn_cache=self_attn_cache)

        attn_values, attn_weights, enc_attn_cache = self.ctx_attn(query, enc_output, mask=dec_enc_attn_mask,
                                                                  enc_attn_cache=enc_attn_cache)

        output = self.pos_ffn(attn_values)

        return output, attn_weights, self_attn_cache, enc_attn_cache


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None, dropout=0.1,
            positional_embedding="sin", layer_norm_first=True, padding_idx=PAD, ffn_activation="relu"):

        super(Decoder, self).__init__()

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model
        self.layer_norm_first = layer_norm_first

        self.embeddings = Embeddings(n_tgt_vocab, d_word_vec,
                                     dropout=dropout,
                                     positional_embedding=positional_embedding,
                                     padding_idx=padding_idx
                                     )

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                         dim_per_head=dim_per_head, layer_norm_first=layer_norm_first, ffn_activation=ffn_activation)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

        self._dim_per_head = dim_per_head

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask, enc_attn_caches=None, self_attn_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        if not self.layer_norm_first:
            emb = self.layer_norm(emb)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(PAD).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache \
                = self.layer_stack[i](output,
                                      enc_output,
                                      dec_slf_attn_mask,
                                      dec_enc_attn_mask,
                                      enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                                      self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None)

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        if self.layer_norm_first:
            output = self.layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1, add_bias=False):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = Linear(self.hidden_size, self.n_words, bias=add_bias)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

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
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class Transformer(NMTModel):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
            dropout=0.1, tie_input_output_embedding=True, tie_source_target_embedding=False, padding_idx=PAD,
            layer_norm_first=True, positional_embedding="sin", generator_bias=False, ffn_activation="relu", **kwargs):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_src_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head,
            padding_idx=padding_idx, layer_norm_first=layer_norm_first, positional_embedding=positional_embedding,
            ffn_activation=ffn_activation)

        self.decoder = Decoder(
            n_tgt_vocab, n_layers=n_layers, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,
            d_inner_hid=d_inner_hid, dropout=dropout, dim_per_head=dim_per_head,
            padding_idx=padding_idx, layer_norm_first=layer_norm_first, positional_embedding=positional_embedding,
            ffn_activation=ffn_activation)

        self.dropout = nn.Dropout(dropout)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        if tie_source_target_embedding:
            assert n_src_vocab == n_tgt_vocab, \
                "source and target vocabulary should have equal size when tying source&target embedding"
            self.encoder.embeddings.embeddings.weight = self.decoder.embeddings.embeddings.weight

        if tie_input_output_embedding:
            self.generator = Generator(n_words=n_tgt_vocab,
                                       hidden_size=d_word_vec,
                                       shared_weight=self.decoder.embeddings.embeddings.weight,
                                       padding_idx=PAD, add_bias=generator_bias)

        else:
            self.generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD,
                                       add_bias=generator_bias)

    def forward(self, src_seq, tgt_seq, log_probs=True):

        enc_output, enc_mask = self.encoder(src_seq)
        dec_output, _, _ = self.decoder(tgt_seq, enc_output, enc_mask)

        return self.generator(dec_output, log_probs=log_probs)

    def encode(self, src_seq):

        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None
        }

    def decode(self, tgt_seq, dec_states, log_probs=True):

        ctx = dec_states["ctx"]
        ctx_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        slf_attn_caches = dec_states['slf_attn_caches']

        dec_output, slf_attn_caches, enc_attn_caches = self.decoder(tgt_seq=tgt_seq, enc_output=ctx, enc_mask=ctx_mask,
                                                                    enc_attn_caches=enc_attn_caches,
                                                                    self_attn_caches=slf_attn_caches)

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)

        dec_states['enc_attn_caches'] = enc_attn_caches
        dec_states['slf_attn_caches'] = slf_attn_caches

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, batch_size, beam_size):

        slf_attn_caches = dec_states['slf_attn_caches']

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        slf_attn_caches = nest.map_structure(
            lambda t: tensor_gather_helper(gather_indices=new_beam_indices,
                                           gather_from=t,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head]),
            slf_attn_caches)

        dec_states['slf_attn_caches'] = slf_attn_caches

        return dec_states


def transformer_base_v1(configs):
    """ Configuration of transformer_base_v1

    This is equivalent to `transformer_base_v1` in tensor2tensor
    """
    # model configurations
    model_configs = configs.setdefault("model_configs", {})
    model_configs['model'] = "Transformer"
    model_configs['n_layers'] = 6
    model_configs['n_head'] = 8
    model_configs['d_model'] = 512
    model_configs['d_word_vec'] = 512
    model_configs['d_inner_hid'] = 2048
    model_configs['dropout'] = 0.1
    model_configs['label_smoothing'] = 0.1
    model_configs["layer_norm_first"] = False
    model_configs['tie_input_output_embedding'] = True

    # optimizer_configs
    optimizer_configs = configs.setdefault("optimizer_configs", {})
    optimizer_configs['optimizer'] = "adam"
    optimizer_configs['learning_rate'] = 0.1
    optimizer_configs['grad_clip'] = - 1.0
    optimizer_configs['optimizer_params'] = {"betas": [0.9, 0.98], "eps": 1e-9}
    optimizer_configs['schedule_method'] = "noam"
    optimizer_configs['scheduler_configs'] = {"d_model": 512, "warmup_steps": 4000}

    return configs


def transformer_base_v2(configs):
    """ Configuration of transformer_base_v2

    This is equivalent to `transformer_base_v2` in tensor2tensor
    """
    configs = transformer_base_v1(configs)

    # model configurations
    model_configs = configs['model_configs']
    model_configs['layer_norm_first'] = True

    # optimizer_configs
    optimizer_configs = configs['optimizer_configs']
    optimizer_configs['learning_rate'] = 0.2
    optimizer_configs['scheduler_configs'] = {"d_model": 512, "warmup_steps": 8000}

    return configs


def transformer_low_resource(configs):
    """ Configuration for training transformer on low-resource datasets.

    This is equivalent to configuration of IWSLT'14 De2en in fairseq.
    """

    configs = transformer_base_v2(configs)

    # model configurations
    model_configs = configs['model_configs']
    model_configs['dropout'] = 0.3

    # optimizer_configs
    optimizer_configs = configs['optimizer_configs']
    optimizer_configs['optimizer'] = "adamw"
    optimizer_configs['learning_rate'] = 0.0005
    optimizer_configs['optimizer_params'] = {"betas": [0.9, 0.98], "eps": 1e-9, "weight_decay": 0.0001}
    optimizer_configs['schedule_method'] = "isqrt"
    optimizer_configs['scheduler_configs'] = {
        "min_lr": 1e-9,
        "warmup_steps": 4000,
        "decay_steps": 4000,
    }

    return configs
