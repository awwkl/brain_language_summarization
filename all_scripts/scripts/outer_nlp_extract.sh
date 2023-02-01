#!/bin/bash

# Change these
nlp_model=led-booksum
seq_len=400
cuda_devices="1"
checkpoint_num=99

# Automatically done
inner_script="./all_scripts/scripts/inner_nlp_extract.sh ${nlp_model} ${seq_len} ${cuda_devices} ${checkpoint_num}"
tmux_session_name="${nlp_model}_len_${seq_len}"
echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

tmux new-session -d -s "${tmux_session_name}"
tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
