import os

import unittests.test_utils as test_utils
from src.utils.logging import INFO


def test_transformer_train(test_dir, use_gpu=False):
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


def test_transformer_inference(test_dir, use_gpu=False):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittests/configs/test_transformer.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = test_utils.get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX + ".final")
    source_path = "./unittests/data/dev/zh.0"
    batch_size = 3
    beam_size = 3
    alpha = 0.6

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=False,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20,
                  alpha=alpha)


def test_transformer_greedy_search(test_dir, use_gpu=False):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittests/configs/test_transformer.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = test_utils.get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX + ".final")
    source_path = "./unittests/data/dev/zh.0"
    batch_size = 3
    beam_size = 1
    alpha = 0.6

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=False,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=10,
                  alpha=alpha)


def test_transformer_ensemble_inference(test_dir, use_gpu=False):
    from src.bin import ensemble_translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittests/configs/test_transformer.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = test_utils.get_model_name(config_path)

    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX + ".final")
    model_path = [model_path for _ in range(3)]

    source_path = "./unittests/data/dev/zh.0"
    batch_size = 3
    beam_size = 3
    alpha = 0.6

    ensemble_translate.run(model_name=model_name,
                           source_path=source_path,
                           batch_size=batch_size,
                           beam_size=beam_size,
                           model_path=model_path,
                           use_gpu=False,
                           config_path=config_path,
                           saveto=saveto,
                           max_steps=20, alpha=alpha)


if __name__ == '__main__':

    test_dir = "./tmp"

    parser = test_utils.build_test_argparser()
    args = parser.parse_args()

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    INFO("=" * 20)
    INFO("Test transformer training...")
    test_transformer_train(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test resuming from training...")
    test_transformer_train(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test transformer inference...")
    test_transformer_inference(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test transformer greedy search...")
    test_transformer_greedy_search(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test ensemble inference...")
    test_transformer_ensemble_inference(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    test_utils.rm_tmp_dir(test_dir)
