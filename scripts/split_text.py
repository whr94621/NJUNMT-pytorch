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
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Split a text file into multiple shards by specific number of lines.")

    parser.add_argument("-f", "--file", type=str,
                        help="""Input file.""")

    parser.add_argument("-n", "--num_lines", type=int,
                        help="""Maximum lines in each shard of the input file.""")

    args = parser.parse_args()

    return args


def main(args):
    idx = 0
    n_lines = 0
    curr_handler = None

    saveto = os.path.basename(args.file)

    with open(args.file) as f:
        for line in f:
            # update handler
            if curr_handler is None or n_lines >= args.num_lines:
                if curr_handler is not None:
                    curr_handler.close()
                curr_handler = open(saveto + ".{0}".format(idx), 'w')
                idx += 1
                n_lines = 0

            curr_handler.write(line)
            n_lines += 1

    curr_handler.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
