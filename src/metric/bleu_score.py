import argparse
import sys

# This is a hack that enables us to use this script independent to NJUNMT-pytorch

try:
    from .bleu_scorer import ExternalScriptBLEUScorer
except Exception:
    from bleu_scorer import ExternalScriptBLEUScorer


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                    help="""
                    Input file.
                    """)

parser.add_argument("-r", "--reference", type=str,
                    help="Path of references.")

parser.add_argument("-t", "--target", type=str,
                    help="Language of target side.")

parser.add_argument("-s", "--script", type=str,
                    help="""
                    Perl script used to evaluate BLEU scores. Can only be within [multi-bleu|multi-bleu-detok].
                    """)

parser.add_argument("-p", "--postprocess", action="store_true",
                    help="""
                    Do post-processing. Here we use 'detruecase.perl' and 'detokenizer.perl' in Moses.
                    """
                    )

parser.add_argument("-d", "--digits_only", action="store_true",
                    help="""
                    Only return digits of BLEU.
                    """)

parser.add_argument("-l", "--lowercase", action="store_true",
                    help="""
                    Lowercase references. This is equivalent to '-lc' option of 'multi-bleu.perl' or 'multi-bleu-detok.perl'.
                    """)

def main(FLAGS):


    scorer = ExternalScriptBLEUScorer(reference_path=FLAGS.reference,
                                      lang=FLAGS.target,
                                      bleu_script=FLAGS.script,
                                      digits_only=FLAGS.digits_only,
                                      lc=FLAGS.lowercase,
                                      postprocess=FLAGS.postprocess)

    print(scorer.corpus_bleu(FLAGS.input))


args = parser.parse_args()
main(args)