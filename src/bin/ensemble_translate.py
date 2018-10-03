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

import argparse

from src.main import ensemble_translate

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str,
                    help="""Name of the model.""")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--model_path", type=str, nargs="+",
                    help="""Path to model files.""")

parser.add_argument("--config_path", type=str,
                    help="""Path to config file.""")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--beam_size", type=int, default=5,
                    help="""Beam size.""")

parser.add_argument("--saveto", type=str,
                    help="""Result prefix.""")

parser.add_argument("--keep_n", type=int, default=-1,
                    help="""To keep how many results. This number should not exceed beam size.""")

parser.add_argument("--use_gpu", action="store_true")

parser.add_argument("--max_steps", type=int, default=150,
                    help="""Max steps of decoding. Default is 150.""")

parser.add_argument("--alpha", type=float, default=-1.0,
                    help="""Factor to do length penalty. Negative value means close length penalty.""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    ensemble_translate(args)


if __name__ == '__main__':
    run()
