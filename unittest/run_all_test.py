import os
import subprocess

# configure_list = [
#     "./unittest/configs/test_transformer.yaml"
# ]

def get_model_name(path):

    return os.path.basename(path).strip().split(".")[0]

def clean_tmp_dir(path):
    subprocess.run("rm -rf {0}".format(path), shell=True)

def test_transformer(test_dir):
    from src.bin import train

    config_path = "./unittest/configs/test_transformer.yaml"
    model_name = get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.test(model_name=model_name,
               config_path=config_path,
               saveto=saveto,
               log_path=log_path,
               valid_path=valid_path,
               debug=True)

    clean_tmp_dir(test_dir)

def test_all():

    test_dir = "./tmp"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    # 1. test transformer
    test_transformer(test_dir)


if __name__ == "__main__":
    test_all()
