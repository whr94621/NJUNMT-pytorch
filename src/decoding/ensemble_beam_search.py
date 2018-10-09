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

from typing import List

import torch

from src.data.vocabulary import BOS, EOS, PAD
from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper


def ensemble_beam_search(nmt_models: List[NMTModel], beam_size, max_steps, src_seqs, alpha=-1.0):
    """

    Args:
        nmt_models (NMTModel):
        beam_size (int):
        max_steps (int):
        src_seqs (torch.Tensor):

    Returns:

    """
    batch_size = src_seqs.size(0)
    num_models = len(nmt_models)

    enc_outputs_group = [nmt_model.encode(src_seqs) for nmt_model in nmt_models]
    init_dec_states_group = [nmt_model.init_decoder(enc_outputs_group[ii], expand_size=beam_size) for ii, nmt_model in
                             enumerate(nmt_models)]

    # Prepare for beam searching
    beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
    final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
    beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
    final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(BOS)

    dec_states_group = init_dec_states_group

    for t in range(max_steps):
        next_probs_group_ = []
        dec_states_group_ = []

        for ii in range(num_models):
            next_probs, dec_states = nmt_models[ii].decode(final_word_indices.view(batch_size * beam_size, -1),
                                                           dec_states_group[ii], log_probs=False)
            next_probs_group_.append(next_probs)
            dec_states_group_.append(dec_states)

        next_scores = torch.log(sum(next_probs_group_) / len(next_probs_group_))
        next_scores = - next_scores  # convert to negative log_probs
        next_scores = next_scores.view(batch_size, beam_size, -1)
        next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

        beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

        vocab_size = beam_scores.size(-1)

        if t == 0 and beam_size > 1:
            # Force to select first beam at step 0
            beam_scores[:, 1:, :] = float('inf')

        # Length penalty
        if alpha > 0.0:
            normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(2) ** alpha
        else:
            normed_scores = beam_scores.detach().clone()

        normed_scores = normed_scores.view(batch_size, -1)

        # Get topK with beams
        _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=False)
        next_beam_ids = torch.div(indices, vocab_size)
        next_word_ids = indices % vocab_size

        # Re-arrange by new beam indices
        beam_scores = beam_scores.view(batch_size, -1)
        beam_scores = torch.gather(beam_scores, 1, indices)

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

        final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=final_lengths,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])

        dec_states_group = [nmt_models[ii].reorder_dec_states(dec_states_group[ii], new_beam_indices=next_beam_ids,
                                                              beam_size=beam_size) for ii in range(num_models)]

        # dec_states = nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids, beam_size=beam_size)

        # If next_word_ids is EOS, beam_mask_ should be 0.0
        beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
        next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                   PAD)  # If last step a EOS is already generated, we replace the last token as PAD
        beam_mask = beam_mask * beam_mask_

        # # If an EOS or PAD is encountered, set the beam mask to 0.0
        final_lengths += beam_mask

        final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)

        if beam_mask.eq(0.0).all():
            break

    # Length penalty
    if alpha > 0.0:
        scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
    else:
        scores = beam_scores / final_lengths

    _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

    return tensor_gather_helper(gather_indices=reranked_ids,
                                gather_from=final_word_indices[:, :, 1:].contiguous(),
                                batch_size=batch_size,
                                beam_size=beam_size,
                                gather_shape=[batch_size * beam_size, -1])
