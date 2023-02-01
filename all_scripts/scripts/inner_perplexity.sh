#!/bin/bash
set -x      # echo on

# Manually change these variables
EXPT_NAME=$1
NLP_MODEL=$2
CUDA_DEVICES=$3

# No need to touch, these are auto-set
OUTPUT_DIR="4-perplexity-results/${EXPT_NAME}/${NLP_MODEL}/"
mkdir -p ${OUTPUT_DIR}

# Calculate perplexity
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}    \
    python calculate_perplexity.py      \
    --nlp_model ${NLP_MODEL}            \
    --output_dir ${OUTPUT_DIR}
