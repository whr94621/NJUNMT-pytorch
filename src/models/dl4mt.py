import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils.common_utils import Vocab
import src.utils.init as my_init
from src.modules.embeddings import Embeddings
from src.modules.cgru import CGRUCell
from src.utils.beam_search import tile_batch, tensor_gather_helper, mask_scores

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

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          bidirectional=True)

        self.linear_bridge = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):

        for weight in self.gru.parameters():
            my_init.rnn_init(weight.data)

        my_init.default_init(self.linear_bridge.weight)

    def bridge(self, context, mask):
        """
        :param context: Context tensor
            with shape [batch_size, ctx_len,dim_ctx]
        :param mask: Mask of context tensor
            with shape [batch_size, ctx_len]
        """

        no_pad_mask = Variable((1.0 - mask.float()))

        ctx_mean = (context * no_pad_mask.unsqueeze(2)).sum(1) / no_pad_mask.unsqueeze(2).sum(1)

        return F.tanh(self.linear_bridge(ctx_mean))


    def forward(self, x):
        """
        :param x: Input sequence.
            with shape [batch_size, seq_len, input_size]
        """
        x_mask = x.data.eq(Vocab.PAD)

        emb = self.embedding(x)

        ctx, _ = self.gru(emb)

        dec_init = self.bridge(ctx, x_mask)

        return ctx, dec_init, x_mask


class Decoder(nn.Module):

    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 dropout_rate=0.0):

        super(Decoder, self).__init__()

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

    def _reset_parameters(self):

        my_init.default_init(self.linear_input.weight)
        my_init.default_init(self.linear_hidden.weight)
        my_init.default_init(self.linear_ctx.weight)

    def forward(self, y, context, context_mask, hidden, one_step=False, cache=None):

        emb = self.embedding(y) # [seq_len, batch_size, dim]

        if one_step:
            (out, attn), hidden = self.cgru_cell(emb, hidden, context, context_mask, cache)
        else:
            # emb: [seq_len, batch_size, dim]
            out = []
            attn = []

            for emb_t in torch.split(emb, split_size=1, dim=0):

                (out_t, attn_t), hidden = self.cgru_cell(emb_t.squeeze(0), hidden, context, context_mask, cache)

                out += [out_t]
                attn += [attn_t]

            out = torch.stack(out)
            attn = torch.stack(attn)

        logits = self.linear_input(emb) + self.linear_hidden(out) + self.linear_ctx(attn)

        logits = F.tanh(logits)

        logits = self.dropout(logits) # [seq_len, batch_size, dim]

        return logits, hidden

class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):

        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)
        self.actn = nn.LogSoftmax(dim=-1)

        if shared_weight is not None:
            self.proj.weight = shared_weight
        else:
            self._reset_parameters()

    def _reset_parameters(self):

        my_init.default_init(self.proj.weight)


    def forward(self, input):
        """
        input == > Linear == > LogSoftmax
        """
        return self.actn(self.proj(input))

class DL4MT(nn.Module):

    def __init__(self, n_src_vocab, n_tgt_vocab, d_word_vec, d_model, dropout, proj_share_weight, **kwargs):

        super().__init__()

        self.encoder = Encoder(n_words=n_src_vocab, input_size=d_word_vec, hidden_size=d_model)

        self.decoder = Decoder(n_words=n_tgt_vocab, input_size=d_word_vec, hidden_size=d_model,
                               dropout_rate=dropout)

        if proj_share_weight is False:
            generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=Vocab.PAD)
        else:
            generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=Vocab.PAD,
                                  shared_weight=self.decoder.embedding.embeddings.weight
                                  )
        self.generator = generator

    def force_teaching(self, x, y):

        ctx, dec_init, ctx_mask = self.encoder(x)

        dec_cache = self.decoder.cgru_cell.compute_cache(ctx)

        logits, _ = self.decoder(y,
                                 context=ctx,
                                 context_mask=ctx_mask,
                                 one_step=False,
                                 hidden=dec_init,
                                 cache=dec_cache) # [tgt_len, batch_size, dim]

        return logits.transpose(1, 0).contiguous() # Convert to batch-first mode.

    def batch_beam_search(self, x, beam_size=5, max_steps=150):

        batch_size = x.size(0)

        ctx, dec_init, ctx_mask = self.encoder(x)
        dec_cache = self.decoder.cgru_cell.compute_cache(ctx)

        ctx = tile_batch(ctx, multiplier=beam_size, batch_dim=0)
        dec_cache = tile_batch(dec_cache, multiplier=beam_size, batch_dim=0)
        hiddens = tile_batch(dec_init, multiplier=beam_size, batch_dim=0).data
        ctx_mask = tile_batch(ctx_mask, multiplier=beam_size, batch_dim=0)

        beam_mask = ctx_mask.new(batch_size, beam_size).fill_(1).float()
        dec_memory_len = ctx_mask.new(batch_size, beam_size).zero_().float()
        beam_scores = ctx_mask.new(batch_size, beam_size).zero_().float()
        final_word_indices = x.data.new(batch_size, beam_size, 1).fill_(Vocab.BOS)

        for t in range(max_steps):

            logits, hiddens = self.decoder(y=Variable(final_word_indices[:,:,-1].contiguous().view(batch_size * beam_size, ), volatile=True),
                                           hidden=Variable(hiddens.view(batch_size * beam_size, -1), volatile=True),
                                           context=ctx,
                                           context_mask=ctx_mask,
                                           one_step=True,
                                           cache=dec_cache
                                           )

            hiddens = hiddens.view(batch_size, beam_size, -1).data

            next_scores = - self.generator(logits)  # [B * Bm, N]
            next_scores = next_scores.view(batch_size, beam_size, -1).data
            next_scores = mask_scores(next_scores, beam_mask=beam_mask)

            beam_scores = next_scores + beam_scores.unsqueeze(2) # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

            vocab_size = beam_scores.size(-1)
            if t == 0:
                beam_scores = beam_scores[:,0,:].contiguous()

            beam_scores = beam_scores.view(batch_size, -1)

            # Get topK with beams
            beam_scores, indices = torch.topk(beam_scores, k=beam_size, dim=-1, largest=False, sorted=False)
            next_beam_ids = torch.div(indices, vocab_size)
            next_word_ids = indices % vocab_size

            # gather beam cache
            dec_memory_len = tensor_gather_helper(gather_indices=next_beam_ids,
                                                  gather_from=dec_memory_len,
                                                  batch_size=batch_size,
                                                  beam_size=beam_size,
                                                  gather_shape=[-1])

            hiddens = tensor_gather_helper(gather_indices=next_beam_ids,
                                           gather_from=hiddens,
                                           batch_size=batch_size,
                                           beam_size=beam_size,
                                           gather_shape=[batch_size * beam_size, -1])

            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=beam_mask,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=final_word_indices,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size,
                                                      gather_shape=[batch_size * beam_size, -1])

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(Vocab.EOS).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0), Vocab.PAD) # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # update beam
            dec_memory_len += beam_mask

            final_word_indices = torch.cat((final_word_indices, torch.unsqueeze(next_word_ids, 2)), dim=2)

            if beam_mask.eq(0.0).all():
                # All the beam is finished (be zero
                break

        scores = beam_scores / (dec_memory_len + 1e-2)

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

        return tensor_gather_helper(gather_indices=reranked_ids,
                                    gather_from=final_word_indices[:, :, 1:].contiguous(),
                                    batch_size=batch_size,
                                    beam_size=beam_size,
                                    gather_shape=[batch_size * beam_size, -1])

    def forward(self, src_seq, tgt_seq=None, mode="train", **kwargs):

        if mode == "train":
            assert tgt_seq is not None

            tgt_seq = tgt_seq.transpose(1, 0).contiguous() # length first

            return self.force_teaching(src_seq, tgt_seq)

        elif mode == "infer":
            return self.batch_beam_search(x=src_seq, **kwargs)