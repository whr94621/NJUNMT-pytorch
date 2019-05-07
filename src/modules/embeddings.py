import math

import torch
import torch.nn as nn

def sinusoidal_positional_embedding(x, min_timescale=1.0, max_timescale=1.0e4):
    batch, length, channels = list(x.size())
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1.))

    position = torch.arange(0, length).to(x)
    inv_timescales = torch.arange(0, num_timescales).to(x)

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1).expand(
        length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
    # scaled time is now length x num_timescales
    # length x channels
    signal = torch.cat([scaled_time.sin(), scaled_time.cos()], 1)

    return signal.unsqueeze(0).expand(batch, length, channels)


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, maximum_position=512):
        super().__init__()

        self.maximum_position = maximum_position
        self.embeddings = nn.Embedding(num_embeddings=maximum_position, embedding_dim=embedding_dim)

    def forward(self, input):
        """

        Args:
            input (torch.Tensor): Input tensor. ([batch_size, seq_len, dim])

        Returns:

        """
        seq_len = input.size(1)
        positions = input.data.new(seq_len).long()
        torch.arange(seq_len, out=positions).unsqueeze(0)  # [1, seq_len]

        return self.embeddings(positions).expand_as(input)


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, min_timescale=1.0, max_timescale=1.0e4):
        super().__init__()

        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, input):
        """

        Args:
            input (torch.Tensor): Input tensor. ([batch_size, seq_len, dim])

        Returns:

        """

        return sinusoidal_positional_embedding(input, self.min_timescale, self.max_timescale)


class Embeddings(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 dropout=0.0,
                 positional_embedding="none",
                 padding_idx=-1):

        super().__init__()

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.padding_idx = padding_idx

        self.embeddings = nn.Embedding(num_embeddings=num_embeddings,
                                       embedding_dim=embedding_dim,
                                       padding_idx=self.padding_idx)

        if positional_embedding == "none":
            self.positional_embeddings = None
        elif positional_embedding == "sin":
            self.positional_embeddings = SinusoidalPositionalEmbedding()
        elif positional_embedding == "learned":
            self.positional_embeddings = LearnedPositionalEmbedding(embedding_dim=embedding_dim)
        else:
            raise ValueError("Unknown positional embedding type {0}".format(positional_embedding))

        self.scale = embedding_dim ** 0.5

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.uniform_(self.embeddings.weight, - 1.0 / self.scale, 1.0 / self.scale)

        with torch.no_grad():
            self.embeddings.weight[self.padding_idx].fill_(0.0)

    def forward(self, x):

        emb = self.embeddings(x)
        emb = emb * self.scale

        if self.positional_embeddings is not None:
            emb += self.positional_embeddings(emb)

        if self.dropout is not None:
            emb = self.dropout(emb)

        return emb
