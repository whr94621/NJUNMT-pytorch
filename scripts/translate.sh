#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

N=$1

export MODEL_NAME="transformer"

python src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path "/home/weihr/NMT_DATA_PY3/1.34M/unittest/MT0$1/src.txt" \
    --model_path "./save/$MODEL_NAME.best.tpz" \
    --config_path "./configs.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "./result/$MODEL_NAME.MT0$1.txt" \
    --use_gpu