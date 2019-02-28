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

import os

import torch
import torch.distributed as dist

MB = 1024 * 1024
BROADCAST_BUCKET_SIZE = 250 * MB
DEFAULT_SHARED_DIR = "/tmp"

__all__ = [
    'distributed_init',
]


def _get_default_group():
    return dist.group.WORLD


def _dist_broadcast_coalesced(tensors, buffer_size=None, process_group=None):
    if process_group is None:
        process_group = dist.distributed_c10d._get_default_group()

    if buffer_size is None:
        buffer_size = BROADCAST_BUCKET_SIZE

    dist._dist_broadcast_coalesced(process_group, tensors, buffer_size, False)


def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for CUDA device without sepecified index, i.e.,
    ``torch.devie('cuda')``, this will return the current default CUDA device if
    :attr:`optional` is ``True``.

    If :attr:`device` is a Python interger, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch.device):
        dev_type = device.type
        if device.type != 'cuda':
            raise ValueError('Expected a cuda device, but got: {}'.format(device))
        device_idx = device.index
    else:
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch.cuda.current_device()
        else:
            raise ValueError('Expected a cuda device with sepecified index or '
                             'an integer, but got: '.format(device))
    return device_idx


def distributed_init(shared_fs_path=None):
    global DEFAULT_SHARED_DIR
    init_method = 'tcp://{address}:{port}'.format(address=os.environ["MASTER_ADDR"], port=os.environ['MASTER_PORT'])
    world_size = int(os.environ['WORLD_SIZE'])

    if world_size == 1:
        raise ValueError('Cannot initialize distributed with distributed_world_size=1')

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        init_method=init_method,
        rank=int(os.environ['RANK'])
    )
    if shared_fs_path is not None:
        DEFAULT_SHARED_DIR = shared_fs_path


from .optimizer import *
from .comm import *
