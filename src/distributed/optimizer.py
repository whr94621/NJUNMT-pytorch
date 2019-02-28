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

# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Below is a modified version of DistributedOptimizer
# In order to manipulate gradients more flexibly,
# we allreduce gradients right before update parameters.
# This may cause performance degradation as we do not
# overlap computation and communication by using hook.

import collections

import torch
import torch.distributed as dist

from src.optim import Optimizer
from . import MB, _get_device_index
from .comm import get_world_size, broadcast_parameters

__all__ = [
    'broadcast_optimizer_state',
    'DistributedOptimizer'
]


class DistributedOptimizer(Optimizer):

    def __init__(self,
                 name,
                 model,
                 lr=0,
                 weight_decay=0,
                 grad_clip=None,
                 optim_args=None,
                 device_id=0,
                 bucket_cap_mb=25,
                 process_group=None,
                 **kwargs):

        super().__init__(name=name, model=model, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip,
                         optim_args=optim_args, distributed=False, **kwargs)

        self.device_ids = [device_id, ]
        self.device_ids = list(map(lambda x: _get_device_index(x, True), self.device_ids))
        self.bucket_cap_mb = bucket_cap_mb
        self.bucket_bytes_cap = bucket_cap_mb * MB

        if process_group is None:
            self.process_group = dist.distributed_c10d._get_default_group()
        else:
            self.process_group = process_group

        self.world_size = get_world_size()
        self._ddp_init_helper()

    def _ddp_init_helper(self):

        self.modules_params_data = [[p.data for p in self.params], ]

        param_buckets = []

        # Split the parameters into buckets and by types as well
        # We only need to bucket and reduce parameters that require grad and
        # this is also true for backward since only the backward hooks for
        # parameters that require grad will be registered with gradient
        # reduction functions
        params_to_bucket = [[], ]
        for p in self.params:
            if p.requires_grad:
                params_to_bucket[0].append(p)

        param_buckets = [dist._dist_bucket_tensors(dev_params_to_bucket,
                                                   int(self.bucket_bytes_cap),
                                                   fine_grained=False)
                         for dev_params_to_bucket in params_to_bucket]

        self.bucket_sizes = []
        self.bucket_map = {}

        for bucket_idx, param_buckets_tuple in enumerate(zip(*param_buckets)):
            self.bucket_sizes.append(0)
            # Now, we transpose again, so we iterate over bucket_elems, but getting tuples
            # of params from each device.
            for param_tuple in zip(*param_buckets_tuple):
                if not param_tuple[0].requires_grad:
                    continue
                for p in param_tuple:
                    self.bucket_map[p] = (bucket_idx, self.bucket_sizes[bucket_idx])
                self.bucket_sizes[bucket_idx] += 1

        self.buckets = [[[None for _ in range(self.bucket_sizes[i])]
                         for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
        self.buckets_ready_size = [[0 for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]

        # coalesced bucket for only device 0
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        # We will always reduce the bucket following the reverse order
        # that is, alway reduces following the order of: n - 1, n - 2, ..., 0
        self.next_bucket = len(self.bucket_sizes) - 1
        # When all buckets are reduced, this will be set to True. This flag is
        # useful for sanity checks to ensure that each iteration's backward has
        # always reduced all buckets
        self.all_buckets_reduced = False
        self.check_previous_reduction = False
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        # default stream tracking to launch nccl reduce kernels
        self.default_streams = []

        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                self.default_streams.append(torch.cuda.current_stream())

    def _queue_reduction(self, bucket_idx):
        # _queue_reduction will use a seperate CUDA stream to coalesce
        # the small tensors to achieve more parallelisms, before passing the
        # coalesced tensor into the c10d CUDA stream for reduction
        result = dist._queue_reduction(self.process_group,
                                       self.buckets[bucket_idx],
                                       self.device_ids)
        self.reduction_works[bucket_idx] = result[0]
        self.buckets_coalesced[bucket_idx] = result[1]

    def allreduce_grad_op(self, param, device_idx):

        bucket_idx, bucket_offset = self.bucket_map[param]

        bucket = self.buckets[bucket_idx][device_idx]
        bucket[bucket_offset] = param.grad.data
        self.buckets_ready_size[bucket_idx][device_idx] += 1

        if device_idx > 0:
            param.grad = None
            param.data.set_()

        if self.buckets_ready_size[bucket_idx][device_idx] == self.bucket_sizes[bucket_idx]:
            self.devs_ready[bucket_idx] += 1

            if self.devs_ready[bucket_idx] < len(self.device_ids):
                return

            # Now all devices's buckets with index: bucket_idx are ready
            if bucket_idx == self.next_bucket:
                self._queue_reduction(bucket_idx)
                self.next_bucket -= 1
                # Now reduce anything that is ready but not yet reduced
                if len(self.ready_buckets_not_reduced) > 0:
                    sorted_todo = sorted(self.ready_buckets_not_reduced, reverse=True)
                    for i in sorted_todo:
                        # Nothing can be reduced now
                        if i < self.next_bucket:
                            break
                        self._queue_reduction(i)
                        self.ready_buckets_not_reduced.remove(i)
                        if i == self.next_bucket:
                            self.next_bucket -= 1
            else:
                self.ready_buckets_not_reduced.add(bucket_idx)

            # When all devices' buckets
            if self.next_bucket == -1:
                # A final sync for all the reduction works
                self._sync_reduction_works()
                self.all_buckets_reduced = True

    def _sync_reduction_works(self):
        # Now only work on the first GPU of self.device_ids
        # _sync_reduction will use a seperate CUDA stream to uncoalesce
        # the coalesced tensors to achieve more parallelisms
        for bucket_idx, grads_batch in enumerate(self.buckets):
            dist._sync_reduction(self.reduction_works[bucket_idx],
                                 grads_batch[0],
                                 self.buckets_coalesced[bucket_idx])

        # Reset the module states
        self.next_bucket = len(self.bucket_sizes) - 1
        self.ready_buckets_not_reduced = set()
        self.reduction_works = [None for _ in range(len(self.bucket_sizes))]
        self.devs_ready = [0 for _ in range(len(self.bucket_sizes))]

        self.buckets = [[[None for _ in range(self.bucket_sizes[i])]
                         for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]
        self.buckets_coalesced = [[] for _ in range(len(self.bucket_sizes))]
        self.buckets_ready_size = [[0 for _ in range(len(self.device_ids))] for i in range(len(self.bucket_sizes))]

    def allreduce_grad(self):
        for param in self.params:
            if param.requires_grad:
                self.allreduce_grad_op(param=param, device_idx=0)

    def step(self, denom=1.0, closure=None):
        self.allreduce_grad()
        return super().step(denom=denom / self.world_size, closure=closure)


def broadcast_optimizer_state(optimizer):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()

        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces allreduce on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.

        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()

        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we need to wrap the scalar in a tensor, then
    # broadcast, then update the appropriate value in the state_dict with the
    # new unwrapped scalar value via a callback.
    def _create_callback(pid, name, t, p):
        def _from_tensor():
            state_dict['state'][pid][name] = t(p.cpu().numpy()[0])

        return _from_tensor

    def _create_option_callback(index, option_key, option_tensor, dtypes):
        def _from_tensor():
            optimizer.param_groups[index][option_key] = _recursive_cast(option_tensor.cpu().numpy()[0], dtypes)

        return _from_tensor

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be wrapped in tensors
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value]).cuda()
            callbacks[key] = _create_option_callback(index, option_key, option_tensor, dtypes)
            params.append((key, option_tensor))

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if not torch.is_tensor(p):
                    # Wrap the scalar in a FloatTensor, and remember its type
                    # so we can cast it back after unwrapping
                    t = type(p)
                    p = torch.Tensor([p]).cuda()
                    callbacks[key] = _create_callback(pid, name, t, p)

                params.append((key, p))

    # Synchronized broadcast of all parameters
    broadcast_parameters(params)

    # Post-broadcast clenaup for non-tensor parameters
    for key, p in params:
        if key in callbacks:
            callbacks[key]()
