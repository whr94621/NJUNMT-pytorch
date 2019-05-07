import os
import re
from typing import List
from .bpe import Bpe
import json

RESERVED_TOKENS_DICT = {
    "<PAD>": (0, 0),
    "<EOS>": (1, 0),
    "<BOS>": (2, 0),
    "<UNK>": (3, 0)
}


class Vocabulary(object):

    def __init__(self, dictionary: dict, max_n_words=-1):
        super().__init__()

        self._token2id_feq = RESERVED_TOKENS_DICT.copy()

        n_reserved_tokens = len(self._token2id_feq)

        for item in dictionary.items():
            self._token2id_feq[item[0]] = (item[1][0] + n_reserved_tokens, item[1][1])

        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])
        self.max_n_words = len(self._token2id_feq) if max_n_words == -1 else max_n_words

    @staticmethod
    def build_from_file(type: str, dict_path: str, max_n_words: int, **kwargs) -> 'Vocabulary':

        if dict_path.endswith(".json"):
            with open(dict_path) as f:
                dictionary = json.load(f)
        else:
            dictionary = dict()
            with open(dict_path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    dictionary[ww] = (i, 0)

        if type == "word":
            return WordVocabulary(dictionary=dictionary, max_n_words=max_n_words)
        elif type == "bpe":
            assert "codes" in kwargs
            assert os.path.exists(kwargs['codes'])

            return BPEVocabulary(dictionary=dictionary, max_n_words=max_n_words, codes=kwargs['codes'])
        else:
            raise ValueError("Unknown vocabulary type {0}".format(type))

    def tokenize(self, sent: str) -> List[str]:
        raise NotImplementedError

    def detokenize(self, tokens: List[str]) -> str:
        raise NotImplementedError

    @property
    def pad(self):
        return 0

    @property
    def eos(self):
        return 1

    @property
    def bos(self):
        return 2

    @property
    def unk(self):
        return 3

    def token2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:

            return self._token2id_feq[word][0]
        else:
            return self.unk

    def id2token(self, word_id):

        return self._id2token[word_id]

    def sent2ids(self, sent):

        tokens = self.tokenize(sent)

        return [self.token2id(t) for t in tokens]

    def ids2sent(self, indices):

        tokens = [self.id2token(ii) for ii in indices if ii not in {self.eos, self.bos, self.pad}]

        return self.detokenize(tokens)


class WordVocabulary(Vocabulary):

    def tokenize(self, sent: str) -> List[str]:
        return sent.strip().split()

    def detokenize(self, tokens: List[str]) -> str:
        return ' '.join(tokens)


class BPEVocabulary(Vocabulary):

    def __init__(self, dictionary: dict, codes: str, max_n_words=-1):
        super().__init__(dictionary=dictionary, max_n_words=max_n_words)
        self.bpe = Bpe(codes=codes)

    def tokenize(self, sent: str) -> List[str]:
        return sum([self.bpe.segment_word(w) for w in sent.strip().split()], [])

    def detokenize(self, tokens: List[str]) -> str:
        return re.sub(r"@@\s|@@$", "", " ".join(tokens))
