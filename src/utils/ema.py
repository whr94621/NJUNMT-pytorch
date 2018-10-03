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

from collections import OrderedDict


class ExponentialMovingAverage(object):

    def __init__(self, named_params, decay, zero_debias=False):

        self._decay = decay
        self._zero_debias = zero_debias

        self._named_params = OrderedDict()
        self._named_params_ave = OrderedDict()

        for name, param in named_params:
            self._named_params[name] = param
            # Add ema
            self._named_params_ave[name] = param.data.clone()

    def step(self):
        if self._decay > 0.0:
            for name, param in self._named_params.items():
                self._named_params_ave[name].sub_((1.0 - self._decay) * (self._named_params_ave[name] - param))

    def state_dict(self):

        state_dict = OrderedDict()

        for name, param in self._named_params_ave.items():
            state_dict[name] = param.data

        return state_dict

    def load_state_dict(self, state_dict):

        for name, param in state_dict.items():

            if name in self._named_params_ave:
                self._named_params_ave[name].copy_(param.data)
