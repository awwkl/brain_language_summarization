#!/bin/bash

# Change these
expt_name="HF_booksum"
nlp_model=led-booksum
checkpoint_num=0
CUDA_DEVICES=1

# Automatically set
CHECKPOINT_DIR="5-finetune-booksum/${expt_name}/${nlp_model}/checkpoint-${checkpoint_num}"
OUTPUT_DIR="6-rouge-score/${expt_name}/${nlp_model}/checkpoint-${checkpoint_num}"

inner_script="CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python ./compute_rouge.py \
                --nlp_model ${nlp_model} --checkpoint_dir ${CHECKPOINT_DIR} --output_dir ${OUTPUT_DIR}"
tmux_session_name="compute_rouge_${expt_name}_${nlp_model}_checkpoint_${checkpoint_num}"
echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

tmux new-session -d -s "${tmux_session_name}"
tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
