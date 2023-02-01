#!/bin/bash

discourse_features=(Full)
subjects=(F H I J K L M N)
ttest_alternatives=(two-sided)

for DISCOURSE_FEATURE in "${discourse_features[@]}"
do
    for SUBJECT in "${subjects[@]}"
    do
        for TTEST_ALTERNATIVE in "${ttest_alternatives[@]}"
        do
            inner_script="python identify_sig_voxels_booksum_minus_base.py --discourse_feature ${DISCOURSE_FEATURE} --subject ${SUBJECT} --ttest_alternative ${TTEST_ALTERNATIVE}"
            tmux_session_name="identify_sig_vox_${DISCOURSE_FEATURE}_${SUBJECT}_${TTEST_ALTERNATIVE}"
            echo "Run: ${inner_script}, tmux: ${tmux_session_name}"

            tmux new-session -d -s "${tmux_session_name}"
            tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

            tmux send-keys -t "${tmux_session_name}" "${inner_script}" Enter
        done
    done
done
