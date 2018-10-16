import os
import subprocess
import argparse


def build_test_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true", help="Running test on GPU")

    return parser

def get_model_name(path):
    return os.path.basename(path).strip().split(".")[0]


def clean_tmp_dir(path):
    subprocess.run("rm -rf {0}/*".format(path), shell=True)


def rm_tmp_dir(path):
    subprocess.run("rm -rf {0}".format(path), shell=True)
