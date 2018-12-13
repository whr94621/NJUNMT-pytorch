#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME="hybrid-nist_zh2en_bpe-base"
CONFIG_PATH=./configs/hybrid_nist_zh2en_bpe_base.yaml
LOG_PATH=/home/user_data/zhengzx/.pytorch.log/${MODEL_NAME}
SAVETO=./save/

python -m src.bin.train \
    --model_name ${MODEL_NAME} \
    --config_path ${CONFIG_PATH} \
    --log_path ${LOG_PATH} \
    --saveto ${SAVETO} \
    --reload \
    --use_gpu