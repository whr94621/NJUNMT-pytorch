#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES='3'

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python ../train.py \
    --model_name "transformer" \
    --reload \
    --config_path "../configs/transformer_adafactor_config.yaml" \
    --log_path "./log_adafactor" \
    --saveto "./save_adafactor/" \
    --use_gpu 
