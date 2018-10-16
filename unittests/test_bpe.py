# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from src.utils.logging import INFO
import unittests.test_utils as test_utils


def test_transformer_train(test_dir, use_gpu=False):
    from src.bin import train

    config_path = "./unittests/configs/test_bpe.yaml"
    model_name = test_utils.get_model_name(config_path)

    saveto = os.path.join(test_dir, "save")
    log_path = os.path.join(test_dir, "log")
    valid_path = os.path.join(test_dir, "valid")

    train.run(model_name=model_name,
              config_path=config_path,
              saveto=saveto,
              log_path=log_path,
              valid_path=valid_path,
              debug=True, use_gpu=use_gpu)


def test_transformer_inference(test_dir, use_gpu=False):
    from src.bin import translate
    from src.utils.common_utils import GlobalNames
    config_path = "./unittests/configs/test_bpe.yaml"

    saveto = os.path.join(test_dir, "save")
    model_name = test_utils.get_model_name(config_path)
    model_path = os.path.join(saveto, model_name + GlobalNames.MY_BEST_MODEL_SUFFIX + ".final")
    source_path = "./unittests/data/dev/zh.0"
    batch_size = 3
    beam_size = 3

    translate.run(model_name=model_name,
                  source_path=source_path,
                  batch_size=batch_size,
                  beam_size=beam_size,
                  model_path=model_path,
                  use_gpu=use_gpu,
                  config_path=config_path,
                  saveto=saveto,
                  max_steps=20)


if __name__ == '__main__':

    parser = test_utils.build_test_argparser()
    args = parser.parse_args()

    test_dir = "./tmp"

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    INFO("=" * 20)
    INFO("Test training with BPE...")
    test_transformer_train(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    INFO("=" * 20)
    INFO("Test inference with BPE...")
    test_transformer_inference(test_dir, use_gpu=args.use_gpu)
    INFO("Done.")
    INFO("=" * 20)

    test_utils.rm_tmp_dir(test_dir)
