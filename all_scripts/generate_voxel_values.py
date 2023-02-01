import os
import numpy as np
import argparse
import pickle

np.random.seed(42)  # Set random seed

from scipy.stats import zscore
def pearson_corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def extract_all_voxels_pearson(pred_path, discourse_feature):
    loaded = np.load(pred_path, allow_pickle=True)
    preds_t_per_feat = loaded.item()['preds_t']                             # (1191, ~27905)
    test_t_per_feat = loaded.item()['test_t']
    
    TR_one_hot_pkl_path = f'./data/TR_one_hot_for_features/{discourse_feature}.pkl'
    TR_one_hot = pickle.load(open(TR_one_hot_pkl_path, 'rb'))
    TR_indices = np.where(TR_one_hot==1)[0]                                 # List of TR indices: [10, 13,...]
    TR_indices = np.random.choice(TR_indices, size=162, replace=False)      # Randomly select 162 TR indices, with set seed
    TR_indices = np.sort(TR_indices)
    
    preds_t_per_feat = preds_t_per_feat[TR_indices, :]                      # Extract from only the desired TRs
    test_t_per_feat =  test_t_per_feat[TR_indices, :]

    return pearson_corr(preds_t_per_feat, test_t_per_feat)                  # (~27905,)

# === Change these variables ===
nlp_model_list = []
# nlp_model_list += ['bart-base', 'bart-booksum']
# nlp_model_list += ['led-base', 'led-booksum']
# nlp_model_list += ['bigbird-base', 'bigbird-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']

# seq_len_list = [20]
# layer_list = [6]
# subject_list = ['F']

seq_len_list = [20, 100, 200, 500]
seq_len_list += [700, 1000]
layer_list = [6, 7, 8, 9, 10, 11, 12]
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

# === Automatically set ===
nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()
base_model_name = nlp_model_list[0].replace('-base', '').replace('-booksum', '')
# base_model_list = [name for name in nlp_model_list if name.endswith('-base')]
# booksum_model_list = [name for name in nlp_model_list if name.endswith('-booksum')]

parser = argparse.ArgumentParser()
parser.add_argument("--discourse_feature", required=True)
parser.add_argument("--subject", required=True)

args = parser.parse_args()
print(args)

discourse_feature = args.discourse_feature
subject = args.subject

print(f'--- Entered: {discourse_feature}, {subject}, models: {nlp_model_list} ---')
base_models_pearson_voxels_list = []
booksum_models_pearson_voxels_list = []

for nlp_model in nlp_model_list:
    if nlp_model.endswith('-booksum'):
        base_or_booksum_voxels_list_to_append = booksum_models_pearson_voxels_list
    else:
        base_or_booksum_voxels_list_to_append = base_models_pearson_voxels_list

    for layer in layer_list:
        for seq_len in seq_len_list:
            pred_path = f'2-encoding_predictions/{nlp_model}/predict_{subject}_with_{nlp_model}_layer_{layer}_len_{seq_len}.npy'
            if not os.path.isfile(pred_path):
                print(f'@@@ Prediction file not found: subject_{subject}_{nlp_model}_layer_{layer}_len_{seq_len}')
                continue
            
            all_voxels_pearson = extract_all_voxels_pearson(pred_path, discourse_feature)
            base_or_booksum_voxels_list_to_append.append(all_voxels_pearson)

base_pearson_voxels_stacked = np.vstack(base_models_pearson_voxels_list)
booksum_pearson_voxels_stacked = np.vstack(booksum_models_pearson_voxels_list)
                                    
base_pearson_voxels = base_pearson_voxels_stacked.mean(0)
booksum_pearson_voxels = booksum_pearson_voxels_stacked.mean(0)
booksum_minus_base_pearson_voxels = booksum_pearson_voxels - base_pearson_voxels

output_pkl_dict = {'base_pearson_voxels': base_pearson_voxels, 'booksum_pearson_voxels': booksum_pearson_voxels,
                    'booksum_minus_base_pearson_voxels': booksum_minus_base_pearson_voxels}
output_pkl_path = f'9-pearson-voxels-for-brain-plot/voxel_values/{base_model_name}/{discourse_feature}/pearson_voxels_{discourse_feature}_{subject}.pkl'
os.makedirs( os.path.dirname(output_pkl_path), exist_ok=True )
with open(output_pkl_path, 'wb') as handle:
    pickle.dump(output_pkl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
