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
import pickle
import uuid

import torch
import torch.distributed as dist

from . import DEFAULT_SHARED_DIR, _get_default_group, _dist_broadcast_coalesced

__all__ = [
    'broadcast_py',
    'broadcast_parameters',
    'all_reduce',
    'all_reduce_py',
    'all_gather_py',
    'all_gather_py_with_shared_fs',
    'broadcast',
    'barrier',
    'get_local_rank',
    'get_world_size',
    'get_rank'
]


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def all_reduce(tensor, group=None):
    if group is None:
        group = _get_default_group()
    return dist.all_reduce(tensor, group=group)


def broadcast(tensor, root_rank=0, group=None):
    if group is None:
        group = _get_default_group()

    dist.broadcast(tensor, root_rank, group=group)


def barrier(group=None):
    if group is None:
        group = _get_default_group()

    dist.barrier(group=group)


def all_reduce_py(data, max_size=65000, average=False, group=None):
    """Apply allreduce to python objects"""
    all_gathered_data = all_gather_py(data=data, max_size=max_size, group=group)

    result = sum(all_gathered_data)

    if not average:
        return result
    else:
        return result / all(all_gathered_data)


def all_gather_py(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_py, '_buffer') or \
            all_gather_py._buffer.numel() < buffer_size:
        all_gather_py._buffer = torch.cuda.ByteTensor(buffer_size)
    buffer = all_gather_py._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255 * 256

    buffer_rank = buffer[rank * max_size: (rank + 1) * max_size]
    buffer_rank[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_rank[1] = enc_size % 255
    buffer_rank[2:enc_size + 2] = torch.ByteTensor(list(enc))

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = (255 * item(out_buffer[0])) + item(out_buffer[1])
            if size > 0:
                result.append(
                    pickle.loads(bytes(out_buffer[2:size + 2].tolist()))
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )


def broadcast_py(data, root_rank=0, max_size=65000):
    """Apply broadcast to python objects"""

    # 1. allocate buffer
    buffer_size = max_size
    if not hasattr(broadcast_py, '_buffer') or \
            broadcast_py._buffer.numel() < buffer_size:
        broadcast_py._buffer = torch.cuda.ByteTensor(buffer_size)

    buffer = broadcast_py._buffer
    buffer.zero_()

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + 2 > max_size:
        raise ValueError('encoded data exceeds max_size: {}'.format(enc_size + 2))

    buffer_broadcasted = buffer
    buffer_broadcasted[0] = enc_size // 255  # this encoding works for max_size < 65k
    buffer_broadcasted[1] = enc_size % 255
    buffer_broadcasted[2:enc_size + 2] = torch.ByteTensor(list(enc))

    broadcast(buffer_broadcasted, root_rank=root_rank)

    size = (255 * buffer_broadcasted[0]) + item(buffer_broadcasted[1])
    obj = pickle.loads(bytes(buffer_broadcasted[2:size + 2].tolist()))

    return obj


def broadcast_parameters(params):
    if isinstance(params, dict):
        params = list(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    _dist_broadcast_coalesced([p for name, p in params])


def gen_random_name():
    """Return a random name for temp file"""
    return uuid.UUID(bytes=os.urandom(16), version=4).hex


def get_local_rank():
    return int(os.environ['LOCAL_RANK'])


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


class _SharedFSTransferProtocol(object):
    """
    Protocol for transfering data between processes by shared filesystem.

    This is useful when you want to transfer some relative big data.
    """

    def __init__(self, prefix="/tmp", name=None):

        self.prefix = prefix

        if name is None:
            name = gen_random_name()

        self.name = name

        self.path = None

    def __getstate__(self):

        return {"prefix": self.prefix, "name": self.name, "path": self.path}

    def __setstate__(self, state):

        self.prefix = state['prefix']
        self.name = state['name']
        self.path = state['path']

    def read(self):

        if self.path is None:
            raise ValueError

        with open(self.path, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def _write(self, obj):

        self.path = os.path.join(self.prefix, self.name) + ".pkl"

        with open(self.path, "wb") as f:
            pickle.dump(obj, f)

    def close(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            # file has been removed by another process
            pass

    @classmethod
    def write(cls, obj, shared_fs_root="/tmp"):

        protoc = cls(prefix=shared_fs_root)
        protoc._write(obj)

        return protoc


def all_gather_py_with_shared_fs(data, root_rank=0):
    tmp_protoc = _SharedFSTransferProtocol.write(data, shared_fs_root=DEFAULT_SHARED_DIR)

    gathered_tmp_protoc = all_gather_py(tmp_protoc)

    gathered_data = [protoc.read() for protoc in gathered_tmp_protoc]

    barrier()

    if get_rank() == root_rank:
        for protoc in gathered_tmp_protoc:
            protoc.close()

    return gathered_data
