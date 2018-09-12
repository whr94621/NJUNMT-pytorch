#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
export MODEL_NAME="transformer-zh2en-word30K"

python3 ./translate.py \
    --model_name $MODEL_NAME \
    --source_path "/home/weihr/NMT_DATA_PY3/1.34M/unittest/MT0$1/src.txt" \
    --model_path "./save/$MODEL_NAME.best.tpz" \
    --config_path "./configs.yaml" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "./result/$MODEL_NAME.MT0$1.txt" \
    --source_bpe_codes "" \
    --use_gpu