#!/bin/bash

nlp_models=(long-t5-booksum)
# seq_lens=(300 400)
# seq_lens=(20 100 200 500)
seq_lens=(20 100 200 500 700 1000)
# seq_lens=(1000)
# layers=(1 2 4 6 8 10 11 12)
layers=(1 2 3 4 5)

for NLP_MODEL in "${nlp_models[@]}"
do
    for SEQ_LEN in "${seq_lens[@]}"
    do
        for LAYER in "${layers[@]}"
        do
            inner_script="./all_scripts/scripts/inner_encoding.sh ${NLP_MODEL} ${SEQ_LEN} ${LAYER}"
            tmux_session_name="${NLP_MODEL}_len_${SEQ_LEN}_layers_${LAYER}"
            echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

            tmux new-session -d -s "${tmux_session_name}"
            tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

            tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
        done
    done
done
