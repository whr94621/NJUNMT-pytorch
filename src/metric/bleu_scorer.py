import os
import subprocess

__all__ = [
    'ExternalScriptBLEUScorer'
]

DETRUECASE_PL = os.path.join(os.path.dirname(__file__), "scripts/recaser/detruecase.perl")
DETOKENIZE_PL = os.path.join(os.path.dirname(__file__), "scripts/tokenizer/detokenizer.perl")
MULTI_BLEU_PL = os.path.join(os.path.dirname(__file__), "scripts/generic/multi-bleu.perl")
MULTI_BLEU_DETOK_PL = os.path.join(os.path.dirname(__file__), "scripts/generic/multi-bleu-detok.perl")
ZH_TOKENIZER_PY = os.path.join(os.path.dirname(__file__), "scripts/tokenizer/tokenizeChinese.py")

class ExternalScriptBLEUScorer(object):
    """Evaluate translation using external scripts.

    Scripts are mainly from moses for post-processing and BLEU computation
    """
    _SCRIPTS = {"multi-bleu": MULTI_BLEU_PL,
                "multi-bleu-detok": MULTI_BLEU_DETOK_PL}


    def __init__(self, reference_path, lang, bleu_script, digits_only=True, lc=False, postprocess=True):
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
        self.reference_path = reference_path
        self.lang = lang.lower()

        if bleu_script not in self._SCRIPTS:
            raise ValueError

        self.script = bleu_script
        self.digits_only = digits_only
        self.lc = lc
        self.postprocess = postprocess

    def _postprocess_cmd(self, stdin):

        # For non-chinese language, we do two-step post-processing
        #   1. detruecase using 'detruecase.perl'
        #   2. detokenize using 'detokenizer.perl'
        if self.lang != "zh":
            cmd_truecase = subprocess.Popen(["perl", DETRUECASE_PL], stdin=stdin, stdout=subprocess.PIPE)
            cmd_postprocess = subprocess.Popen(["perl", DETOKENIZE_PL, "-q", "-u", "-l", self.lang],
                                              stdin=cmd_truecase.stdout,
                                              stdout=subprocess.PIPE)
        else:
            # For chinese, we first remove all the space.
            # Then, we use external scripts to split into chinese characters except those non-chinese words.

            cmd_rm_space = subprocess.Popen(["sed","s/ //g"], stdin=stdin, stdout=subprocess.PIPE)
            cmd_postprocess = subprocess.Popen(["python", ZH_TOKENIZER_PY, "-p"], stdin=cmd_rm_space.stdout,
                                               stdout=subprocess.PIPE)

        return cmd_postprocess

    def _compute_bleu(self, stdin):

        cmd_bleu_str = ["perl",]

        cmd_bleu_str.append(self._SCRIPTS[self.script])

        # Whether convert reference into lower case
        if self.lc:
            cmd_bleu_str.append("-lc")

        cmd_bleu_str.append(self.reference_path)

        cmd_bleu = subprocess.Popen(cmd_bleu_str, stdin=stdin, stdout=subprocess.PIPE)

        cmd_out = cmd_bleu.communicate()[0]

        return cmd_out

    def corpus_bleu(self, hyp_in):

        cmd_bpe = subprocess.Popen(["sed", "s/@@ //g"], stdin=hyp_in, stdout=subprocess.PIPE)

        if self.postprocess:
            cmd_postprocess = self._postprocess_cmd(stdin=cmd_bpe.stdout)
        else:
            cmd_postprocess = cmd_bpe

        out = self._compute_bleu(stdin=cmd_postprocess.stdout).decode("utf-8").strip().split("\n")
        out = [line for line in out if line.startswith("BLEU")][0]

        if self.digits_only:
            out = out.split(',')[0].split("=")[-1].strip()
            out = float(out)

        return out
