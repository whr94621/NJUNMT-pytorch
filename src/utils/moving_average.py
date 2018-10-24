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
from collections import OrderedDict

class MovingAverage(object):
    """
    This is the common moving average formula.

    $$
    p^{ave}_t += (p_t - p^{ave}_{t-1}) * \alpha
    $$

    If $\alpha$ is a fixed value, this is exponential moving average. If $alpha$ is $1/t$,
    this is accumulated moving average.
    """

    def __init__(self, moving_average_method, named_params, alpha=0.0):

        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be a float value between [0.0, 1.0)")

        if moving_average_method not in {"sma", "ema"}:
            raise ValueError("Unknown moving average method {0}".format(moving_average_method))

        self.moving_average_method = moving_average_method
        self.alpha = alpha

        self.named_params_ave = None

        self.num_acc_steps = 1

        self.named_params = OrderedDict()
        self.named_params_ave = OrderedDict()

        for name, param in named_params:
            self.named_params[name] = param
            self.named_params_ave[name] = param.data.clone()

    def step(self):

        if self.moving_average_method == "sma":
            self.num_acc_steps += 1
            alpha = 1.0 / self.num_acc_steps
        else:
            alpha = self.alpha

        with torch.no_grad():
            for name, param in self.named_params.items():
                self.named_params_ave[name].sub_(alpha * (self.named_params_ave[name] - param))

    def export_ma_params(self):

        ma_params = OrderedDict()
        for name, param in self.named_params_ave.items():
            ma_params[name] = param.data

        return ma_params

    def state_dict(self):

        state = dict()

        state["num_acc_steps"] = self.num_acc_steps
        state['ma_params'] = self.export_ma_params()

        return state

    def load_state_dict(self, state_dict):

        self.num_acc_steps = state_dict['num_acc_steps']

        for name, param in state_dict['ma_params'].items():
            if name in self.named_params_ave:
                self.named_params_ave[name].copy_(param.data)
