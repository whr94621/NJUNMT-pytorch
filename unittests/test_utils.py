import os
import subprocess

def get_model_name(path):
    return os.path.basename(path).strip().split(".")[0]

def clean_tmp_dir(path):
    subprocess.run("rm -rf {0}/*".format(path), shell=True)

def rm_tmp_dir(path):
    subprocess.run("rm -rf {0}".format(path), shell=True)