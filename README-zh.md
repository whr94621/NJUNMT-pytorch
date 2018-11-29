# NJUNMT-pytorch

---
[English](README.md), [中文](README-zh.md)
---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.com/whr94621/NJUNMT-pytorch.svg?branch=dev-travis-ci)](https://travis-ci.com/whr94621/NJUNMT-pytorch)

NJUNMT-pytorch是一个开源的神经机器翻译工具包。这个工具包主要是为了方便机器翻译的研究，其中包括了如下一些基线系统：

- [DL4MT-tutorial](https://github.com/nyu-dl/dl4mt-tutorial): 一个被广泛用作基于RNN的神经机器翻译模型的基线系统实现。据我们所指，这是目前唯一的和广为使用的的DL4MT系统相一致的pytorch实现。

- [Attention is all you need](https://arxiv.org/abs/1706.03762): 谷歌提出的一个强大的神经机器翻译模型。这个模型完全依赖于注意力机制构建。

## 目录
- [NJUNMT-pytorch](#njunmt-pytorch)
    - [目录](#%E7%9B%AE%E5%BD%95)
    - [依赖的包](#%E4%BE%9D%E8%B5%96%E7%9A%84%E5%8C%85)
    - [使用说明](#%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E)
        - [快速开始](#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
        - [1. 建立词表](#1-%E5%BB%BA%E7%AB%8B%E8%AF%8D%E8%A1%A8)
        - [2. 修改配置文件](#2-%E4%BF%AE%E6%94%B9%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)
        - [3. 训练](#3-%E8%AE%AD%E7%BB%83)
        - [4. 解码](#4-%E8%A7%A3%E7%A0%81)
    - [Benchmark](#benchmark)
    - [和我们联系](#%E5%92%8C%E6%88%91%E4%BB%AC%E8%81%94%E7%B3%BB)

## 依赖的包

- python 3.5+
- pytorch 0.4.0+
- tqdm
- tensorboardX
- sacrebleu

## 使用说明

### 快速开始
我们提供了一键在NIST中英数据集上训练和解码transformer模型的脚本(只在南京大学自然语言处理组的服务器上可用)。只需要在项目的根目录下执行

``` bash
bash ./scripts/train.sh
```

来进行模型训练，以及执行

``` bash
# 3 means decoding on NIST 2003. This value
# can also be 4,5,6, which represents NIST 2004, 2005, 2006 respectively. 
bash ./scripts/translate.sh 3 
```

在NIST 2003数据集上进行解码。下面我们将详细说明如何配置训练和解码。

### 1. 建立词表

首先我们需要为源端和目标端建立词表文件。我们提供了一个脚本```./data/build_dictionary.py```来建立json格式的词表

请通过运行:

``` bash
python ./scripts/build_dictionary.py --help
```

来查看该脚本的帮助文件。

我们强烈推荐不要在这里限制词表的大小，而是通过模型的配置文件在训练时来设定。

### 2. 修改配置文件

可以参考```./configs```文件夹中的一些样例。我们提供了几种配置样例:

- ```dl4mt_nist_zh2en.yaml```: 在NIST中英上训练一个DL4MT模型
- ```transformer_nist_zh2en.yaml```: 在NIST中英上训练一个词级别的transformer模型
- ```transformer_nist_zh2en_bpe.yaml```: 在NIST中英上训练一个使用BPE的transformer模型
- ```transformer_wmt14_en2de.yaml```: 在WMT14英德上训练一个transformer模型

了解更多关于如何配置一个神经机器翻译模型的训练任务，请参考
[这里](https://github.com/whr94621/NJUNMT-pytorch/wiki/Configuration)。

### 3. 训练

通过运行如下脚本来启动一个训练任务

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

执行```python -m src.bin.train --help```来查看更多的选项

训练期间，所有检查点和最好的模型都会被保存在```---saveto```指定的文件夹下面。假设模型的名称被设定为"MyModel"，那么这个文件夹下面会出现如下一些文件：

- **MyModel.ckpt**: 存放了所有保存的检查点名称的文本文件
- **MyModel.ckpt.xxxx**: 在第xxxx步保存的检查点文件
- **MyModel.best**: 存放了所有最好检查点名称的文本文件
- **MyModel.best.xxxx**: 在第xxxx步保存的检查点文件
- **MyModel.best.final**: 最终得到的最好模型文件, 即在验证集上取得最好效果的模型。其中只保留了模型参数

### 4. 解码

当训练结束时，最好的模型会被自动的保存。通常我们只需要用被命名为"xxxx.best.final"的最好模型文件来进行解码。如之前所说，这个模型能在验证集上取得最好的效果

我们可以通过执行下列脚本来进行解码:

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

通过运行```python -m src.bin.translate --help```来查看更多的选项。

同样我们的代码支持集成解码。通过运行```python -m src.bin.ensemble_translate --help```来查看更多的选项。

## Benchmark

请查看[BENCHMARK.md](./BENCHMARK.md)

## 和我们联系

如果你有任何问题，请联系[whr94621@foxmail.com](mailto:whr94621@foxmail.com)