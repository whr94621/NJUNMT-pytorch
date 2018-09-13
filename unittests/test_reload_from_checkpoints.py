import os
from src.utils.logging import INFO
import unittests.test_utils as test_utils


def test_transformer_train(test_dir):
    from src.bin import train

    config_path = "./unittests/configs/test_transformer.yaml"
    model_name = test_utils.get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
              config_path=config_path,
              saveto=saveto,
              log_path=log_path,
              valid_path=valid_path,
              debug=True)

if __name__ == '__main__':

    test_dir = "./tmp"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    INFO("=" * 20)
    INFO("Test transformer training...")
    test_transformer_train(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test reloading from latest checkpoint...")
    test_transformer_train(test_dir)
    INFO("Done.")
    INFO("=" * 20)

    test_utils.rm_tmp_dir(test_dir)
