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

import torch
from horovod.torch import mpi_ops
from horovod.torch.compression import Compression

__all__ = [
    'DistributedOptimizer'
]


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression):

        super(self.__class__, self).__init__(params)
        self._compression = compression

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        self._parameter_names = {v: k for k, v
                                 in sorted(named_parameters)}

        self._handles = {}
        self._grad_accs = []

    def _allreduce_grad(self, p):
        name = self._parameter_names.get(p)
        tensor = p.grad.data
        tensor_compressed, ctx = self._compression.compress(tensor)
        # We use sum here and manually rescaled gradient by passing a denominator to optimizer.
        handle = mpi_ops.allreduce_async_(tensor_compressed, average=False, name=name)
        return handle, ctx

    def synchronize(self):
        ctx = None
        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._allreduce_grad(p)
                self._handles[p] = (handle, ctx)
        for p, (handle, _) in self._handles.items():
            output = mpi_ops.synchronize(handle)
            p.grad.data.set_(self._compression.decompress(output, ctx))
        self._handles.clear()

    def step(self, closure=None):
        return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.
    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression)
