import os
import subprocess
from src.utils.logging import INFO

# configure_list = [
#     "./unittest/configs/test_transformer.yaml"
# ]

def get_model_name(path):
    return os.path.basename(path).strip().split(".")[0]

def clean_tmp_dir(path):
    subprocess.run("rm -rf {0}/*".format(path), shell=True)

def rm_tmp_dir(path):
    subprocess.run("rm -rf {0}".format(path), shell=True)

def test_transformer_train(test_dir):
    from src.bin import train

    config_path = "./unittest/configs/test_transformer.yaml"
    model_name = get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
               config_path=config_path,
               saveto=saveto,
               log_path=log_path,
               valid_path=valid_path,
               debug=True)

def test_transformer_inference(test_dir):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittest/configs/test_transformer.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    source_path = "./unittest/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=False,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20)

def test_dl4mt_train(test_dir):

    from src.bin import train

    config_path = "./unittest/configs/test_dl4mt.yaml"
    model_name = get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
               config_path=config_path,
               saveto=saveto,
               log_path=log_path,
               valid_path=valid_path,
               debug=True)

def test_dl4mt_inference(test_dir):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames

    config_path = "./unittest/configs/test_dl4mt.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX)
    source_path = "./unittest/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=False,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20)

def test_all():

    test_dir = "./tmp"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    INFO("=" * 20)
    INFO("Test transformer training...")
    test_transformer_train(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test transformer inference...")
    test_transformer_inference(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    clean_tmp_dir(test_dir)

    INFO("=" * 20)
    INFO("Test DL4MT training...")
    test_dl4mt_train(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test DL4MT inference...")
    test_dl4mt_inference(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    rm_tmp_dir(test_dir)

if __name__ == "__main__":
    test_all()
