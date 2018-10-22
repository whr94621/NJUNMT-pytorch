# NJUNMT-pytorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.com/whr94621/NJUNMT-pytorch.svg?branch=dev-travis-ci)](https://travis-ci.com/whr94621/NJUNMT-pytorch)

NJUNMT-pytorch is an open-source toolkit for neural machine translation.
This toolkit is highly research-oriented, which contains some common baseline
model:

- [DL4MT-tutorial](https://github.com/nyu-dl/dl4mt-tutorial): A rnn-base nmt model widely used as baseline. To our knowledge, this is the
only pytorch implementation which is exactly the same as original model.([nmtpytorch](https://github.com/lium-lst/nmtpytorch) is another pytorch implementation but with minor structure difference.)

- [Attention is all you need](https://arxiv.org/abs/1706.03762): A strong nmt model introduced by Google, which only relies on attenion
mechanism.


# Requirements

- python 3.5+
- pytorch 0.4.0+
- tqdm
- tensorboardX
- sacrebleu

# Usage

## 1. Build Vocabulary
First we should generate vocabulary files for both source and 
target language. We provide a script in ```./data/build_dictionary.py``` to build them in json format.

See how to use this script by running:
``` bash
python ./scripts/build_dictionary.py --help
```
We highly recommend not to set the limitation of the number of
words and control it by config files while training.

## 2. Write Configuration File

See examples in ```./configs``` folder.  We provide several examples:

- ```dl4mt_nist_zh2en.yaml```: to run a DL4MT model on NIST Chinese to Enligsh
- ```transformer_nist_zh2en.yaml```: to run a Transformer model on NIST Chinese to English
- ```transformer_nist_zh2en_bpe.yaml```: to run a Transformer model on NIST Chinese to English using BPE.
- ```transformer_wmt14_en2de.yaml```: to run a Transformer model on WMT14 English to German

To further learn how to configure a NMT training task, see [this](https://github.com/whr94621/NJUNMT-pytorch/wiki/Configuration) wiki page.

## 3. Training
We can setup a training task by running
``` bash
export CUDA_VISIBLE_DEVICES=0
python -m src.bin.train \
    --model_name <your-model-name> \
    --reload \
    --config_path <your-config-path> \
    --log_path <your-log-path> \
    --saveto <path-to-save-checkpoints> \
    --valid_path <path-to-save-validation-translation> \
    --use_gpu
```
See detail options by running ```python -m src.bin.train --help```.

## Translation

When training is over, our code will automatically save the best model. We can translation any text by running:
``` bash
export CUDA_VISIBLE_DEVICES=0
python -m src.bin.translate \
    --model_name <your-model-name> \
    --source_path <path-to-source-text> \
    --model_path <path-to-model> \
    --config_path <path-to-configuration> \
    --batch_size <your-batch-size> \
    --beam_size <your-beam-size> \
    --alpha <your-length-penalty> \
    --use_gpu
```
See detail options by running ```python -m src.bin.translate --help```.

Also our code support ensemble decoding. See more options by running ```python -m src.bin.ensemble_translate --help```

# Benchmark

See [BENCHMARK.md](./BENCHMARK.md)

# Acknowledgement

- This code is heavily borrowed from OpenNMT/OpenNMT-py and have been
simplified for research use.

# Contact
If you have any question, please contact [whr94621@foxmail.com](mailto:whr94621@foxmail.com)




