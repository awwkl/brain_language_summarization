# CUDA_VISIBLE_DEVICES=0                  \
#     python -m debugpy --listen 5678 --wait-for-client     \
#     test-booksum.py

# Get NLP features (representations from NLP model)
    # python -m debugpy --listen 5678 --wait-for-client     \
    #                                     \
# CUDA_VISIBLE_DEVICES=7                  \
#     python \
#     extract_nlp_features.py             \
#     --nlp_model longformer              \
#     --sequence_length 80                \
#     --output_dir 1-nlp_features/longformer
    
# Build encoding model to predict fMRI recordings
# CUDA_VISIBLE_DEVICES=5                  \
    # python -m debugpy --listen 5678 --wait-for-client     \
    #                                     \
    # predict_brain_from_nlp.py           \
    # --subject F                         \
    # --nlp_feat_type led-booksum                \
    # --nlp_feat_dir 1-nlp_features/led-booksum/         \
    # --layer 8                           \
    # --sequence_length 20                \
    # --output_dir 2-encoding_predictions/  \

# Evaluate predictions of encoding model using classification accuracy
CUDA_VISIBLE_DEVICES=5                      \
    python -m debugpy --listen 5678 --wait-for-client     \
                                            \
    evaluate_brain_predictions.py           \
    --input_path 2-encoding_predictions/bart-base/predict_F_with_bart-base_layer_6_len_20.npy    \
    --output_path 3-eval-results/old/predict_F_with_bert_layer_8_len_20           \
    --subject F



# Without debugger
# CUDA_VISIBLE_DEVICES=5                  \
#     python extract_nlp_features.py      \
#     --nlp_model bert                    \
#     --sequence_length 20                \
#     --output_dir nlp_features