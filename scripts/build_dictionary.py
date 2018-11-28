#!/usr/bin/python
# author: Hao-Ran Wei
# email: whr94621@gmail.com

"""
Build dictionary from corpus.

output: word X word_freq, descent ordered by word frequency
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict


def INFO(string):
    time_format = '%Y-%m-%d %H:%M:%S'
    sys.stderr.write('{0}: {1}\n'.format(time.strftime(time_format), string))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--num', type=int, default=0, help="Number of words to keep.")
    parser.add_argument('--freq', type=int, default=0, help="Least frequency to keep")
    parser.add_argument('--char', action='store_true', default=False, help="Split words into characters.")
    parser.add_argument('--verbose', type=int, default=100000)
    return parser


def main(filename, num, freq, char, verbose):
    assert num * freq == 0, 'Choose only one between -N and -F'

    # print 'Processing', filename

    INFO('Processing {0}'.format(filename))
    word_freqs = OrderedDict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            words_in = line.strip().split(' ')
            if char:
                words_in = sum([list(w.decode('utf-8')) for w in words_in], [])

            for w in words_in:
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1

            if verbose > 0 and (i + 1) % verbose == 0:
                INFO('Parsed {0} lines'.format(i + 1))

    INFO('Generate vocabulary...')
    word_freqs = [(w, f) for w, f in word_freqs.items()]
    word_freqs = sorted(word_freqs, key=lambda x: -x[1])

    if num != 0 and freq == 0:
        word_freqs = word_freqs[:num - 2]
    elif num == 0 and freq != 0:
        word_freqs = [(w, f) for w, f in word_freqs if f >= freq]
    else:
        INFO('Keep all the tokens!')

    # We only use dict instead of OrderedDict
    # As the token ids are the natural order.
    worddict = OrderedDict()
    n = 0
    for ii, ww in enumerate(word_freqs):
        worddict[ww[0]] = (ii + n, ww[1])

    INFO('{0} words remain'.format(len(worddict)))
    INFO('Least frequency: {0}'.format(word_freqs[-1][1]))
    INFO('Done.')
    INFO('Save at {0}'.format('%s.json' % os.path.basename(filename)))

    with open('%s.json' % os.path.basename(filename), 'w') as f:
        json.dump(worddict, f, indent=1)

    print('Done')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(filename=args.file, num=args.num, freq=args.freq, char=args.char, verbose=args.verbose)
