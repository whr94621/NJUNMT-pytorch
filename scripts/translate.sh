#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

export MODEL_NAME="transformer"

for i in 2 3 4 5 6; 
do 
python ../translate.py \
    --model_name $MODEL_NAME \
    --source_path "/home/user_data/zouw/zh2en_134W/testsets/mt0$i.src.tok" \
    --model_path "./save_adam/$MODEL_NAME.best.tpz" \
    --config_path "../configs/transformer_base_config.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "mt0$i.txt" \
    --keep_n 1 \
	--use_gpu
done;
