#!/bin/bash
# set -x      # echo on

# Use this command to run this script
# nohup ./all_scripts/run_scripts.sh  &> 0-logs/08-16.txt &

# Manually change these variables
NLP_MODEL="bigbird-booksum"

# layers=(24)
# seq_lens=(20 100 200 300 400 500 700 1000)
# subjects=(F H I J K L M N)

layers=(32)
seq_lens=(300 400 1000)
subjects=(F H I J K L M N)

# No need to touch, these are auto-set
NLP_FEAT_DIR="1-nlp_features/${NLP_MODEL}/"
ENCODE_PRED_DIR="2-encoding_predictions/${NLP_MODEL}/"
EVAL_RESULTS_DIR="3-eval-results/${NLP_MODEL}/"
mkdir -p ${NLP_FEAT_DIR} ${ENCODE_PRED_DIR} ${EVAL_RESULTS_DIR}

# (2) Build encoding model to predict fMRI recordings, then (3) Evaluate the predictions
for LAYER in "${layers[@]}"
do
    for SEQ_LEN in "${seq_lens[@]}"
    do
        for SUBJECT in "${subjects[@]}"
        do
            BRAIN_PREDS_PATH="${ENCODE_PRED_DIR}/predict_${SUBJECT}_with_${NLP_MODEL}_layer_${LAYER}_len_${SEQ_LEN}.npy"
            EVAL_RESULTS_PATH="${EVAL_RESULTS_DIR}/predict_${SUBJECT}_with_${NLP_MODEL}_layer_${LAYER}_len_${SEQ_LEN}"

            tmux_session_name="${NLP_MODEL}_len_${SEQ_LEN}_layer_${LAYER}_subject_${SUBJECT}"
            tmux new-session -d -s "${tmux_session_name}"
            echo "tmux: ${tmux_session_name}"
            
            tmux send-keys -t "${tmux_session_name}" "source env/bin/activate" Enter

            tmux send-keys -t "${tmux_session_name}" "  \
                python predict_brain_from_nlp.py        \
                --subject ${SUBJECT}                    \
                --nlp_feat_type ${NLP_MODEL}            \
                --nlp_feat_dir ${NLP_FEAT_DIR}          \
                --layer ${LAYER}                        \
                --sequence_length ${SEQ_LEN}            \
                --output_dir ${ENCODE_PRED_DIR}" Enter

            tmux send-keys -t "${tmux_session_name}" "      \
                python evaluate_brain_predictions.py        \
                    --input_path ${BRAIN_PREDS_PATH}        \
                    --output_path ${EVAL_RESULTS_PATH}      \
                    --subject ${SUBJECT}" Enter

            tmux send-keys -t "${tmux_session_name}" "rm ${BRAIN_PREDS_PATH}" Enter
        done
    done
done
