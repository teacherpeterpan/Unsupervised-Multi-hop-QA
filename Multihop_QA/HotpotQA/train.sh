#!/bin/bash

set -x

DATAHOME=../data/processed_for_qa
MODELHOME=../models_new/unsupervised_paraphrase

mkdir -p ${MODELHOME}

export CUDA_VISIBLE_DEVICES=1

python code/run_mrqa.py \
  --do_train \
  --do_eval \
  --model spanbert-large-cased \
  --train_file ${DATAHOME}/train.top100000.paraphrase.txt \
  --dev_file ${DATAHOME}/dev.human.txt \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_per_epoch 4 \
  --output_dir ${MODELHOME} \
  # --finetuning_dir ../models_new/unsupervised_supervised
  # --gpu_index 3
