import os
import subprocess
from subprocess import DEVNULL

__all__ = [
    'SacreBLEUScorer'
]

DETRUECASE_PL = os.path.join(os.path.dirname(__file__), "scripts/recaser/detruecase.perl")
DETOKENIZE_PL = os.path.join(os.path.dirname(__file__), "scripts/tokenizer/detokenizer.perl")
ZH_TOKENIZER_PY = os.path.join(os.path.dirname(__file__), "scripts/tokenizer/tokenizeChinese.py")


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

        if sacrebleu_args is None:
            self.sacrebleu_args = []
        else:
            self.sacrebleu_args = sacrebleu_args.strip().split()

        if num_refs == 1:
            self.references = [self.reference_path,]
        else:
            self.references = ["{0}{1}".format(self.reference_path, ii) for ii in range(self.num_refs)]

        self.src_lang, self.tgt_lang = self.lang_pair.split("-")[1]
        self.postprocess = postprocess
        self.test_set = test_set


    def _postprocess_cmd(self, stdin):

        cmd_detrucase = subprocess.Popen(["perl", DETRUECASE_PL], stdin=stdin, stdout=subprocess.PIPE, stderr=DEVNULL)
        cmd_postprocess = subprocess.Popen(["perl", DETOKENIZE_PL, "-q", "-l", self.tgt_lang],
                                           stdin=cmd_detrucase.stdout, stdout=subprocess.PIPE, stderr=DEVNULL)
        return cmd_postprocess

    def _compute_bleu(self, stdin):

        sacrebleu_cmd = ["sacrebleu", "-l", self.lang_pair] + self.sacrebleu_args + ["--score-only", ]

        if self.test_set is not None:
            sacrebleu_cmd += ['--test-set', ] + [self.test_set]
        else:
            sacrebleu_cmd += self.references

        cmd_bleu = subprocess.Popen(["sacrebleu", "-l", self.lang_pair] + self.sacrebleu_args + ["--score-only",] + self.references,
                                    stdin=stdin,
                                    stdout=subprocess.PIPE
                                    )

        bleu = cmd_bleu.communicate()[0].decode("utf-8").strip()

        try:
            bleu = float(bleu)
        except:
            print(type(bleu))
            print(bleu)
            exit(1)

        return bleu

    def corpus_bleu(self, hyp_in):

        if self.postprocess:
            cmd_postprocess = self._postprocess_cmd(stdin=hyp_in)
            inp = cmd_postprocess.stdout
        else:
            inp = hyp_in

        bleu = self._compute_bleu(stdin=inp)

        return bleu