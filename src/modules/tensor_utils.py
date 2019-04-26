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
import numpy as np

FLOAT32_INF = float(np.finfo('float32').max / 10)


def tile_batch(x, multiplier, batch_dim=0):
    """Copy several times for all the tensors in a batch

    Given a batch of tensors whose shape is [..., bsz, ...], `tile_batch` copy all the tensors in this batch with
    k times (=> [..., bsz, k, ...]), and then restore its original dimension (=> [..., bsz*k, ...]).

    """
    x_size = x.size()
    out_size = x_size[:batch_dim] + (x_size[batch_dim] * multiplier,) + x_size[batch_dim + 1:]

    x_tiled = torch.unsqueeze(x, dim=batch_dim + 1)
    x_tiled = x_tiled.repeat(*[1 if d != batch_dim + 1 else multiplier for d in range(len(x_size) + 1)])
    x_tiled = x_tiled.view(*out_size)

    return x_tiled
