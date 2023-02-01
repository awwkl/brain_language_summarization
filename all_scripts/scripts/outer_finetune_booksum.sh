#!/bin/bash

# Change these
expt_name="my_finetuned"
nlp_model=long-t5-base

# Automatically set
OUTPUT_DIR="5-finetune-booksum/${expt_name}/${nlp_model}/"

inner_script="python ./finetune_booksum.py --nlp_model ${nlp_model} --output_dir ${OUTPUT_DIR}"
tmux_session_name="finetune_booksum_${expt_name}_${nlp_model}"
echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

tmux new-session -d -s "${tmux_session_name}"
tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
