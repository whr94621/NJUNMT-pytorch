# -*- coding: UTF-8 -*- 

# Copyright 2018, Natural Language Processing Group, Nanjing University, 
#
#       Author: Zheng Zaixiang
#       Contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.models.base import NMTModel
from . import transformer, dl4mt


class TransEncRNNDec(dl4mt.DL4MT, transformer.Transformer):
    def __init__(self, n_src_vocab, n_tgt_vocab, **config):
        super(transformer.Transformer, self).__init__()

        self.config = config

        self.encoder = transformer.Encoder(
            n_src_vocab=n_src_vocab, n_layers=config['n_layers'], n_head=config['n_head'],
            d_word_vec=config['d_word_vec'], d_model=config['d_model'],
            d_inner_hid=config['d_inner_hid'], dropout=config['dropout'])

        self.decoder = dl4mt.Decoder(
            n_words=n_tgt_vocab, input_size=config['d_word_vec'],
            hidden_size=config['d_model'], context_size=config['d_model'],
            dropout_rate=config['dropout'],
            bridge_type=config['bridge_type'])

        self.generator = dl4mt.Generator(n_words=n_tgt_vocab,
                                         hidden_size=config['d_word_vec'], padding_idx=PAD)

        # if config['share_enc_dec_embedding']:
        #     assert config['n_src_vocab'] == config['n_tgt_vocab'], \
        #         "Sharing embeddings of encoder and decoder requires equal size of their vocabularies."
        #     self.decoder.embedding.embeddings.weight = self.encoder.embeddings.embeddings.weight

        if config['proj_share_weight']:
            self.generator.proj.weight = self.decoder.embedding.embeddings.weight

    def encode(self, src_seq):
        return transformer.Transformer.encode(self, src_seq)

    def init_decoder(self, enc_outputs, expand_size=1):
        return dl4mt.DL4MT.init_decoder(self, enc_outputs, expand_size)

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        return dl4mt.DL4MT.reorder_dec_states(self, dec_states, new_beam_indices, beam_size)

    def forward(self, src_seq, tgt_seq, log_probs=True):
        enc_ctx, enc_mask = self.encoder(src_seq)

        dec_init, dec_cache = self.decoder.init_decoder(enc_ctx, enc_mask)

        logits, _ = self.decoder(tgt_seq,
                                 context=enc_ctx,
                                 context_mask=enc_mask,
                                 one_step=False,
                                 hidden=dec_init,
                                 cache=dec_cache)  # [batch_size, tgt_length dim]

        return self.generator(logits, log_probs)

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


