#!/bin/bash
set -x      # echo on

# Use this command to run this script
# nohup ./all_scripts/run_scripts.sh  &> 0-logs/08-16.txt &

# Manually change these variables
NLP_MODEL=$1
SEQ_LEN=$2
CUDA_DEVICES=$3
CHECKPOINT_NUM=$4

# No need to touch, these are auto-set
NLP_FEAT_DIR="1-nlp_features/${NLP_MODEL}/"
ENCODE_PRED_DIR="2-encoding_predictions/${NLP_MODEL}/"
EVAL_RESULTS_DIR="3-eval-results/${NLP_MODEL}/"
mkdir -p ${NLP_FEAT_DIR} ${ENCODE_PRED_DIR} ${EVAL_RESULTS_DIR}

CHECKPOINT_DIR="5-finetune-booksum/my_finetuned/led-base/checkpoint-${CHECKPOINT_NUM}"

# (1) Get NLP features (representations from NLP model)
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}     \
    python extract_nlp_features.py      \
    --nlp_model ${NLP_MODEL}            \
    --sequence_length ${SEQ_LEN}        \
    --output_dir ${NLP_FEAT_DIR}        \
    --checkpoint_dir ${CHECKPOINT_DIR}

