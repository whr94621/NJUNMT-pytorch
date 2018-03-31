import os
from src.utils import batch_open
from .bleu_score import corpus_bleu
import subprocess

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



def _en_de_script(hyp_path, ref_path):

    # Get the dirname of this file
    script_path = os.path.join(os.path.dirname(__file__), "scripts/multi-bleu-detok.perl")
    cmd_cat = subprocess.Popen(["cat", hyp_path], stdout=subprocess.PIPE)
    cmd_bleu = subprocess.Popen(["perl", script_path, ref_path], stdin=cmd_cat.stdout, stdout=subprocess.PIPE)

    out = cmd_bleu.communicate()[0]

    bleu =  out.decode("utf-8").strip().split(',')[0].split("=")[-1].strip()

    return float(bleu)

def _en_zh_script(hyp_path, ref_path):
    # Get the dirname of this file
    ref_path_ = ref_path.rsplit(".", 1)[0] + "."

    script_path = os.path.join(os.path.dirname(__file__), "scripts/multi-bleu.perl")
    cmd_cat = subprocess.Popen(["cat", hyp_path], stdout=subprocess.PIPE)
    cmd_bleu = subprocess.Popen(["perl", script_path, "-lc", ref_path_], stdin=cmd_cat.stdout, stdout=subprocess.PIPE)

    out = cmd_bleu.communicate()[0]

    bleu =  out.decode("utf-8").strip().split(',')[0].split("=")[-1].strip()

    return float(bleu)

EVAL_SCRIPTS = {
    "de-en": _en_de_script,
    "en-de": _en_de_script,
    "en-zh": _en_zh_script,
    "zh-en": _en_zh_script
}

class ExternalScriptBLEUScorer(object):

    def __init__(self, reference_path, lang_pair="en-de"):

        self.reference_path = reference_path

        if lang_pair not in EVAL_SCRIPTS:
            raise ValueError("Not supported language pair {0}".format(lang_pair))

        self.lang_pair = lang_pair

    def corpus_bleu(self, hypothesis_path):

        if not os.path.exists(hypothesis_path):
            raise ValueError("{0} not exists".format(hypothesis_path))


        try:
            bleu = EVAL_SCRIPTS[self.lang_pair](hypothesis_path, self.reference_path)
        except ValueError:
            bleu = 0.0
        return bleu
