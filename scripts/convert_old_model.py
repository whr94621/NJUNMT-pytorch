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


""" Convert parameter's names of previous version into new version
Usage:
    python convert_old_model.py <old-checkpoint-path>
"""

import os
import sys
from collections import OrderedDict

import torch


def convert_transformer_names(name: str):
    old_name = name

    # 1. add transform layer after slf_attn / ctx_attn / pos_ffn
    if "layer_norm" not in name:
        name = name.replace("slf_attn", "slf_attn.transform_layer")
        name = name.replace("ctx_attn", "ctx_attn.transform_layer")
        name = name.replace("pos_ffn", "pos_ffn.transform_layer")
    else:
        if "out_layer_norm" in name:
            name = name.replace("out_layer_norm", "layer_norm")

        # rename layer normalization in decoder block
        if "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "slf_attn.layer_norm")

        if "layer_norm_2" in name:
            name = name.replace("layer_norm_2", "ctx_attn.layer_norm")

        if "encoder.block_stack" in name and not ("slf_attn" in name or "pos_ffn" in name):
            name = name.replace("layer_norm", "slf_attn.layer_norm")

    name = name.replace("block_stack", "layer_stack")

    if old_name != name:
        print("Convert {0} to {1}".format(old_name, name))

    return name


ckpt = torch.load(sys.argv[1], map_location="cpu")  # type: dict

if "model" in ckpt:
    new_ckpt = dict()
    for k, v in ckpt.items():
        if k != "model":
            new_ckpt[k] = v

    new_ckpt['model'] = OrderedDict()
    for k, v in ckpt['model'].items():
        new_ckpt['model'][convert_transformer_names(k)] = v

else:
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        new_ckpt[convert_transformer_names(k)] = v

torch.save(new_ckpt, "new." + os.path.basename(sys.argv[1]))
