#!/bin/bash

discourse_features=(Full)
subjects=(F H I J K L M N)

for DISCOURSE_FEATURE in "${discourse_features[@]}"
do
    for SUBJECT in "${subjects[@]}"
    do
        inner_script="python all_scripts/generate_voxel_values.py --discourse_feature ${DISCOURSE_FEATURE} --subject ${SUBJECT}"
        tmux_session_name="gen_vox_values_${DISCOURSE_FEATURE}_${SUBJECT}"
        echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

        tmux new-session -d -s "${tmux_session_name}"
        tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

        tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
    done
done
