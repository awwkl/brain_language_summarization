#!/bin/bash
set -x      # echo on

# Use this command to run this script
# nohup ./all_scripts/run_scripts.sh  &> 0-logs/08-16.txt &

# Set by command line arguments passed in
# E.g. inner_encoding.sh bigbird-booksum 700 8 10
NLP_MODEL=$1        # 1st argument: bigbird-booksum
SEQ_LEN=$2          # 2nd argument: 700
LAYERS=("${@:3}")   # 3rd + remaining arguments: (8 10)

SUBJECTS=(F H I J K L M N)

# No need to touch, these are auto-set
NLP_FEAT_DIR="1-nlp_features/${NLP_MODEL}/"
ENCODE_PRED_DIR="2-encoding_predictions/${NLP_MODEL}/"
EVAL_RESULTS_DIR="3-eval-results/${NLP_MODEL}/"
mkdir -p ${NLP_FEAT_DIR} ${ENCODE_PRED_DIR} ${EVAL_RESULTS_DIR}

# Telegraph beforehand what commands will be run
for layer in "${LAYERS[@]}"
do
    for subject in "${SUBJECTS[@]}"
    do
        echo "@@@ NLP_model: ${NLP_MODEL}, layer: ${layer}, subject: ${subject}"
    done
done

# (2) Build encoding model to predict fMRI recordings, then (3) Evaluate the predictions
for layer in "${LAYERS[@]}"
do
    for subject in "${SUBJECTS[@]}"
    do
        echo "@@@ NLP_model: ${NLP_MODEL}, layer: ${layer}, subject: ${subject}"
        BRAIN_PREDS_PATH="${ENCODE_PRED_DIR}/predict_${subject}_with_${NLP_MODEL}_layer_${layer}_len_${SEQ_LEN}.npy"
        EVAL_RESULTS_PATH="${EVAL_RESULTS_DIR}/predict_${subject}_with_${NLP_MODEL}_layer_${layer}_len_${SEQ_LEN}"

        echo "--- Build encoding model for subject ${subject} ---"
        python predict_brain_from_nlp.py            \
            --subject ${subject}                    \
            --nlp_feat_type ${NLP_MODEL}            \
            --nlp_feat_dir ${NLP_FEAT_DIR}          \
            --layer ${layer}                        \
            --sequence_length ${SEQ_LEN}            \
            --output_dir ${ENCODE_PRED_DIR}         \

        # echo "--- Evaluate predictions of encoding model for subject ${subject} ---"
        # python evaluate_brain_predictions.py        \
        #     --input_path ${BRAIN_PREDS_PATH}        \
        #     --output_path ${EVAL_RESULTS_PATH}      \
        #     --subject ${subject}                    \
        
        # rm ${BRAIN_PREDS_PATH}
        # echo "completed for layer: ${layer}, subject: ${subject}";
    done
done
