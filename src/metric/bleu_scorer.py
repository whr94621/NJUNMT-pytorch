import argparse
import os
import sacrebleu
from sacremoses import MosesDetokenizer, MosesDetruecaser

__all__ = [
    'SacreBLEUScorer'
]


class SacreBLEUScorer(object):
    """Evaluate translation using external scripts.

    Scripts are mainly from moses for post-processing and BLEU computation
    """

    def __init__(self, reference_path, lang_pair, sacrebleu_args=None, postprocess=False, num_refs=1, test_set=None):
        """Initialize Scorer

        Args:
            reference_path: Path to reference files. If there are multiple reference files, such as
                'ref.0', 'ref.1', ..., 'ref.n', just pass a 'ref.'

            lang: Language of reference such as en, de, zh and et al.

            bleu_script: Script used to calculate BLEU. Only ```multi-bleu.perl``` and ```multi-bleu-detok.perl```
                are supported here.

            digits_only: Return bleu score only. Default is true.

            lc: Whether to use the lowercase of reference. It is equivalent to '-lc' option of the multi-bleu script.

            postprocess: Whether do post-processing.
        """

        self.lang_pair = lang_pair.lower()
        self.reference_path = reference_path
        self.num_refs = num_refs

        parser = argparse.ArgumentParser()
        parser.add_argument('-lc', action='store_true', default=False)
        parser.add_argument('--tokenize', '-tok', choices=sacrebleu.TOKENIZERS.keys(), default='13a')

        if sacrebleu_args is None:
            sacrebleu_args = ""

        self.sacrebleu_args = parser.parse_args(sacrebleu_args.split())

        references = []

        if num_refs == 1:
            with open(self.reference_path) as f:
                references.append(f.readlines())
        else:
            for ii in range(self.num_refs):
                with open("{0}{1}".format(self.reference_path, ii)) as f:
                    references.append(f.readlines())

        self.references = references

        self.src_lang, self.tgt_lang = self.lang_pair.split("-")
        self.postprocess = postprocess

        if self.postprocess:
            self.detokenizer = MosesDetokenizer(lang=self.tgt_lang)
            self.detrucaser = MosesDetruecaser()

        self.test_set = test_set

    def _postprocess(self, string: str):

        string = self.detrucaser.detruecase(string, return_str=False)
        string = self.detokenizer.detokenize(string, return_str=True)

        return string

    def corpus_bleu(self, hyp_in):
        """
        hyp_in should be a list/steam of strings where each line is a hypothesis.
        """

        if self.postprocess:
            hyp_in = [self._postprocess(line.strip()) for line in hyp_in]

        bleu = sacrebleu.corpus_bleu(sys_stream=hyp_in, ref_streams=self.references,
                                     lowercase=self.sacrebleu_args.lc,
                                     tokenize=self.sacrebleu_args.tokenize)

        return bleu.score
