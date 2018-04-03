import sys
import time
import contextlib
import pickle as pkl
import json
import numpy as np
from torch.autograd import Variable


from . import nest
from src.metric.bleu_score import corpus_bleu



__all__ = [
    'batch_open',
    'ERROR',
    'GlobalNames',
    'INFO',
    'PRINT',
    'Timer',
    'WARN',
    'Collections',
    'LearningRateDecay',
    'BLEUScorer',
    'Vocab',
    'sequence_mask',
    'build_vocab_shortlist',
    'to_gpu',
    'to_variable'
]

# ================================================================================== #
# File I/O Utils

@contextlib.contextmanager
def batch_open(refs, mode='r'):
    handlers = []
    if not isinstance(refs, (list, tuple)):
        refs = [refs]
    for f in refs:
        handlers.append(open(f, mode))

    yield handlers

    for h in handlers:
        h.close()


class GlobalNames:
    # learning rate variable name
    MY_LEARNING_RATE_NAME = "learning_rate"

    MY_CHECKPOINIS_PREFIX = ".ckpt"

    MY_BEST_MODEL_SUFFIX = ".best.tpz"

    MY_BEST_OPTIMIZER_PARAMS_SUFFIX = ".best_optim.tpz"

    MY_COLLECTIONS_SUFFIX = ".collections.pkl"

    MY_MODEL_ARCHIVES_SUFFIX = ".archives.pkl"

    USE_GPU = False


time_format = '%Y-%m-%d %H:%M:%S'

class Timer(object):
    def __init__(self):
        self.t0 = 0

    def tic(self):
        self.t0 = time.time()

    def toc(self, format='m:s', return_seconds=False):
        t1 = time.time()

        if return_seconds is True:
            return t1 - self.t0

        if format == 's':
            return '{0:d}'.format(t1 - self.t0)
        m, s = divmod(t1 - self.t0, 60)
        if format == 'm:s':
            return '%d:%02d' % (m, s)
        h, m = divmod(m, 60)
        return '%d:%02d:%02d' % (h, m, s)

def ERROR(string):
    sys.stderr.write('{0} ERROR: {1}\n'.format(time.strftime(time_format), string))

def INFO(string):
    time_format = '%Y-%m-%d %H:%M:%S'
    print('{0}: {1}'.format(time.strftime(time_format), string))

def PRINT(*string):
    ss = [s if isinstance(s, str) else '{0}'.format(s) for s in string]
    sys.stderr.write('{0}\n'.format(' '.join(ss)))


def WARN(string):
    sys.stderr.write('{0} WARNING: {1}\n'.format(time.strftime(time_format), string))

class Collections(object):

    """Collections for logs during training.

    Usually we add loss and valid metrics to some collections after some steps.
    """
    _MY_COLLECTIONS_NAME = "my_collections"

    def __init__(self, kv_stores=None, name=None):

        self._kv_stores = kv_stores if kv_stores is not None else {}

        if name is None:
            name = Collections._MY_COLLECTIONS_NAME
        self._name = name

    def load(self, archives):

        if self._name in archives:
            self._kv_stores = archives[self._name]
        else:
            self._kv_stores = []

    def add_to_collection(self, key, value):
        """
        Add value to collection

        :type key: str
        :param key: Key of the collection

        :param value: The value which is appended to the collection
        """
        if key not in self._kv_stores:
            self._kv_stores[key] = [value]
        else:
            self._kv_stores[key].append(value)

    def export(self):
        return {self._name: self._kv_stores}

    def get_collection(self, key):
        """
        Get the collection given a key

        :type key: str
        :param key: Key of the collection
        """
        if key not in self._kv_stores:
            return []
        else:
            return self._kv_stores[key]
    @staticmethod
    def pickle(path, **kwargs):
        """
        :type path: str
        """
        archives_ = dict([(k,v) for k,v in kwargs.items()])

        if not path.endswith(".pkl"):
            path = path + ".pkl"

        with open(path, 'wb') as f:
            pkl.dump(archives_, f)

    @staticmethod
    def unpickle(path):
        """:type path: str"""

        with open(path, 'rb') as f:
            archives_ = pkl.load(f)

        return archives_

class LearningRateDecay(object):

    def __init__(self, max_patience, min_lrate=5e-5, start_steps=0):
        self._max_patience = max_patience
        self._min_lrate = min_lrate
        self._start_steps = start_steps

        self._min_loss = 1000000.0
        self._bad_counts = 0

    def decay(self, n_steps, loss, lrate):

        if n_steps < self._start_steps:
            return lrate

        if loss < self._min_loss:
            self._min_loss = loss
            self._bad_counts = 0
        else:
            self._bad_counts += 1

            if self._bad_counts >= self._max_patience and lrate > self._min_lrate:
                self._bad_counts = 0
                lrate = lrate / 2.0

            lrate = max(lrate, self._min_lrate) # No less than the minimum learning rate

        return lrate

class BLEUScorer(object):

    def __init__(self, reference_path, use_char=False):

        if not isinstance(reference_path, list):
            raise ValueError("reference_path must be a list")

        self._reference_path = reference_path

        _references = []
        with batch_open(self._reference_path) as fs:
            for lines in zip(*fs):
                if use_char is False:
                    _references.append([line.strip().split() for line in lines])

                else:
                    _references.append([list(line.strip().replace(" ", "")) for line in lines])

        self._references = _references

    def corpus_bleu(self, hypotheses):

        return corpus_bleu(list_of_references=self._references,
                           hypotheses=hypotheses,
                           emulate_multibleu=True)

class Vocab(object):

    PAD = 0
    EOS = 1
    UNK = 3
    BOS = 2

    def __init__(self, dict_path, max_n_words=-1):

        with open(dict_path) as f:
            _dict = json.load(f)

        # Word to word index and word frequence.
        self._token2id_feq = self._init_dict()

        N = len(self._token2id_feq)

        for ww, vv in _dict.items():
            if isinstance(vv, int):
                self._token2id_feq[ww] = (vv + N, 0)
            else:
                self._token2id_feq[ww] = (vv[0] + N, vv[1])

        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])

        self._max_n_words = max_n_words

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<PAD>": (Vocab.PAD, 0),
            "<UNK>": (Vocab.UNK, 0),
            "<EOS>": (Vocab.EOS, 0),
            "<BOS>": (Vocab.BOS, 0)
                }

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return Vocab.UNK

    def id2token(self, id):

        return self._id2token[id]

    @staticmethod
    def special_ids():

        return [0, 1, 2]

def sequence_mask(seqs_length):

    maxlen = np.max(seqs_length)

    row_vector = np.arange(maxlen)

    mask = row_vector[None,:] < np.expand_dims(seqs_length, -1)

    return mask.astype('float32')

def build_vocab_shortlist(shortlist):

    shortlist_ = nest.flatten(shortlist)

    shortlist_ = sorted(list(set(shortlist_)))

    shortlist_np = np.array(shortlist_).astype('int64')

    map_to_shortlist = dict([(wid, sid) for sid, wid in enumerate(shortlist_np)])
    map_from_shortlist = dict([(item[1], item[0]) for item in map_to_shortlist.items()])

    return shortlist_np, map_to_shortlist, map_from_shortlist

def to_gpu(*inputs):

    return list(map(lambda x: x.cuda(), inputs))

def to_variable(*inputs, use_gpu=False, volatile=False):

    if use_gpu:
        outs = to_gpu(*inputs)
    else:
        outs = inputs
    outs = list(map(lambda x: Variable(x.contiguous(), volatile=volatile), outs))

    if len(outs) == 1:
        return outs[0]
    else:
        return outs
    # return list(map(lambda x: Variable(x.contiguous(), requires_grad=False, volatile=volatile), outs))

# OPTIMIZERS = {
#     "adam": torch.optim.Adam,
#     "adadelta": torch.optim.Adadelta,
#     "rmsprop": torch.optim.RMSprop,
#     "sgd": torch.optim.SGD,
# }
#
#
# class Optimizer(object):
#
#     def __init__(self,
#                  name,
#                  params,
#                  lrate,
#                  grad_clip=None,
#                  **kwargs):
#
#         if name not in OPTIMIZERS:
#             raise ValueError("Unknown optimizer %s" % name)
#
#         self.optim = OPTIMIZERS[name](params=params, lr=lrate, **kwargs)
#
#         self.grad_clip = grad_clip
#         self.lrate = lrate
#
#     def _grad_clipping(self):
#
#         if self.grad_clip is not None:
#             for group in self.optim.param_groups:
#                 torch.nn.utils.clip_grad_norm(parameters=group['params'], max_norm=self.grad_clip)
#
#     def step(self, closure=None):
#
#         # 1. clip gradients
#         self._grad_clipping()
#
#         # 2. update
#         self.optim.step(closure=closure)
#
#         # 3. zero gradients
#         self.optim.zero_grad()
#
#     def set_lrate(self, lr):
#
#         for group in self.optim.param_groups:
#             group['lr'] = lr
#
#     def zero_grad(self):
#         self.optim.zero_grad()