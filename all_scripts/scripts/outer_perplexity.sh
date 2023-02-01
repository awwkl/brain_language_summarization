#!/bin/bash

expt_name="whole_early3"
nlp_model=bart-booksum
CUDA_DEVICES="1"

inner_script="./all_scripts/scripts/inner_perplexity.sh ${expt_name} ${nlp_model} ${CUDA_DEVICES}"
tmux_session_name="perplexity_${expt_name}_${nlp_model}"
echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

tmux new-session -d -s "${tmux_session_name}"
tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
