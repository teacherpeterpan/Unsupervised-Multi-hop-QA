#!/bin/bash

set -x

DATAHOME=./data
MODELHOME=./outputs/supervised

mkdir -p ${MODELHOME}

export CUDA_VISIBLE_DEVICES=3

python code/run_mrqa.py \
  --do_train \
  --do_eval \
  --model spanbert-large-cased \
  --train_file ${DATAHOME}/train.human.json \
  --dev_file ${DATAHOME}/dev.human.json \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_per_epoch 10 \
  --output_dir ${MODELHOME} \
