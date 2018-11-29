# NJUNMT-pytorch

---
[English](README.md), [中文](README-zh.md)
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.com/whr94621/NJUNMT-pytorch.svg?branch=dev-travis-ci)](https://travis-ci.com/whr94621/NJUNMT-pytorch)

NJUNMT-pytorch is an open-source toolkit for neural machine translation.
This toolkit is highly research-oriented, which contains some common baseline
model:

- [DL4MT-tutorial](https://github.com/nyu-dl/dl4mt-tutorial): A rnn-base nmt model widely used as baseline. To our knowledge, this is the
only pytorch implementation which is exactly the same as original model.([nmtpytorch](https://github.com/lium-lst/nmtpytorch) is another pytorch implementation but with minor structure difference.)

- [Attention is all you need](https://arxiv.org/abs/1706.03762): A strong nmt model introduced by Google, which only relies on attenion
mechanism.

## Table of Contents
- [NJUNMT-pytorch](#njunmt-pytorch)
    - [Table of Contents](#table-of-contents)
    - [Requirements](#requirements)
    - [Usage](#usage)
        - [0. Quick Start](#0-quick-start)
        - [1. Build Vocabulary](#1-build-vocabulary)
        - [2. Write Configuration File](#2-write-configuration-file)
        - [3. Training](#3-training)
        - [4. Translation](#4-translation)
    - [Benchmark](#benchmark)
    - [Contact](#contact)

## Requirements

- python 3.5+
- pytorch 0.4.0+
- tqdm
- tensorboardX
- sacrebleu

## Usage

### 0. Quick Start

We provide push-button scripts to setup training and inference of
transformer model on NIST Chinese-English Corpus (only on NJUNLP's
server). Just execute under root directory of this repo
``` bash
bash ./scripts/train.sh
```
for training and
``` bash
# 3 means decoding on NIST 2003. This value
# can also be 4,5,6, which represents NIST 2004, 2005, 2006 respectively. 
bash ./scripts/translate.sh 3 
```

### 1. Build Vocabulary
First we should generate vocabulary files for both source and 
target language. We provide a script in ```./data/build_dictionary.py``` to build them in json format.

See how to use this script by running:
``` bash
python ./scripts/build_dictionary.py --help
```
We highly recommend not to set the limitation of the number of
words and control it by config files while training.

### 2. Write Configuration File

See examples in ```./configs``` folder.  We provide several examples:

- ```dl4mt_nist_zh2en.yaml```: to run a DL4MT model on NIST Chinese to Enligsh
- ```transformer_nist_zh2en.yaml```: to run a Transformer model on NIST Chinese to English
- ```transformer_nist_zh2en_bpe.yaml```: to run a Transformer model on NIST Chinese to English using BPE.
- ```transformer_wmt14_en2de.yaml```: to run a Transformer model on WMT14 English to German

To further learn how to configure a NMT training task, see [this](https://github.com/whr94621/NJUNMT-pytorch/wiki/Configuration) wiki page.

### 3. Training
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

During training, checkpoints and best models will be saved under the directory specified by option ```---saveto```. Suppose that the model name is "MyModel", there would be several files under that directory:

- **MyModel.ckpt**: A text file recording names of all the kept checkpoints

- **MyModel.ckpt.xxxx**: Checkpoint stored in step xxxx

- **MyModel.best**: A text file recording names of all the kept best checkpoints
  
- **MyModel.best.xxxx**: Best checkpoint stored in step xxxx.
  
- **MyModel.best.final**: Final best model, i.e., the model achieved best performance on validation set. Only model parameters are kept in it.

### 4. Translation

When training is over, our code will automatically save the best model. Usually you could just use the final best model, which is named as xxxx.best.final, to translate. This model achieves the best performance on the validation set.

We can translation any text by running:

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

## Benchmark

See [BENCHMARK.md](./BENCHMARK.md)

## Contact

If you have any question, please contact [whr94621@foxmail.com](mailto:whr94621@foxmail.com)