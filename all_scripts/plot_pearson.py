import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns

np.random.seed(42)  # Set random seed

from scipy.stats import zscore
def pearson_corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def compute_pearson_from_preds_numpy(pred_path, expt_setting):
    print('compute_pearson_from_preds_numpy:', pred_path, expt_setting)
    loaded = np.load(pred_path, allow_pickle=True)
    preds_t_per_feat = loaded.item()['preds_t']                             # (1191, ~27905)
    test_t_per_feat = loaded.item()['test_t']
    
    TR_one_hot_pkl_path = f'./data/TR_one_hot_for_features/{expt_setting}.pkl'
    TR_one_hot = pickle.load(open(TR_one_hot_pkl_path, 'rb'))
    TR_indices = np.where(TR_one_hot==1)[0]                                 # List of TR indices: [10, 13,...]
    TR_indices = np.random.choice(TR_indices, size=162, replace=False)      # Randomly select 162 TR indices, with set seed
    TR_indices = np.sort(TR_indices)
    
    preds_t_per_feat = preds_t_per_feat[TR_indices, :]                      # Extract from only the desired TRs
    test_t_per_feat =  test_t_per_feat[TR_indices, :]

    pearson_per_voxel = pearson_corr(preds_t_per_feat, test_t_per_feat)     # (~27905,)
    pearson_scalar = np.mean(pearson_per_voxel)                             # 1 scalar number
    return pearson_scalar, pearson_per_voxel

def get_pearson_from_saved_pkl(pkl_path):
    # print('get_pearson_from_saved_pkl:', pkl_path)
    loaded = pickle.load(open(pkl_path, 'rb'))
    return loaded['mean_pearson_across_subjects'], loaded['pearson_dict']

def get_base_model_type(nlp_model):
    for model_type in ['led', 'long-t5', 'bigbird', 'bart']:
        if nlp_model.startswith(model_type):
            return model_type
    return "UNKNOWN_MODEL_TYPE"

def get_base_or_booksum(nlp_model):
    if nlp_model.endswith('base'):
        return 'base'
    if nlp_model.endswith('booksum'):
        return 'booksum'
    return "UNKNOWN_base_or_booksum"

# === Change these variables ===
nlp_model_list = []
nlp_model_list += ['bart-base', 'bart-booksum']
nlp_model_list += ['led-base', 'led-booksum']
nlp_model_list += ['bigbird-base', 'bigbird-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']
# nlp_model_list += ['bart-booksum', 'led-booksum', 'bigbird-booksum', 'long-t5-booksum']

expt_setting_list = ['Characters', 'Emotion', 'Motion']
expt_setting_list +=  ['Non-discourse', 'Full']

nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

seq_len_list = [20, 100, 200, 500, 700, 1000]
# layer_list = [4, 6, 8, 10, 11, 12]
layer_list = [6, 8, 11]
# layer_list = [1, 6, 11]
# layer_list = [11]
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

df_out = pd.DataFrame()
for expt_setting in expt_setting_list:
    for nlp_model in nlp_model_list:
        for layer in layer_list:
            for seq_len in seq_len_list:
                
                # If final Pearson correlation results have been computed and saved, just load it
                pearson_saved_path = f'7-pearson-saved/{expt_setting}/{nlp_model}/pearson_{nlp_model}_layer_{layer}_len_{seq_len}.pkl'
                if os.path.isfile(pearson_saved_path):
                    mean_pearson_across_subjects, pearson_dict = get_pearson_from_saved_pkl(pearson_saved_path)

                # Otherwise, compute the Pearson score for each subject, then save it to a pkl file for future loading
                else:
                    pearson_dict = {}
                    for subject in subject_list:
                        pred_path = f'2-encoding_predictions/{nlp_model}/predict_{subject}_with_{nlp_model}_layer_{layer}_len_{seq_len}.npy'
                        if not os.path.isfile(pred_path):
                            continue
                        
                        try:
                            pearson_scalar, pearson_per_voxel = compute_pearson_from_preds_numpy(pred_path, expt_setting)
                            pearson_dict[subject] = pearson_scalar
                        except Exception:
                            print('@@@ !!! Exception, but no issue -- pickle file in process of being written to')
                    
                    # If number of subjects NOT complete
                    if len(pearson_dict) < len(subject_list):
                        print(f'=== Not done: {nlp_model}_layer_{layer}_len_{seq_len}, # subjects = {len(pearson_dict)} ===')
                        continue
                    else:
                    # If all 8 subjects are in, then Save the results since it is complete
                        mean_pearson_across_subjects = sum(pearson_dict.values()) / len(pearson_dict)
                        mean_pearson_across_subjects = round(mean_pearson_across_subjects, 4)
                        for subj in pearson_dict.keys():
                            pearson_dict[subj] = round(pearson_dict[subj], 4)

                        os.makedirs( os.path.dirname(pearson_saved_path), exist_ok=True )
                        with open(pearson_saved_path, 'wb') as handle:
                            pearson_to_pkl = {'mean_pearson_across_subjects': mean_pearson_across_subjects, 'pearson_dict': pearson_dict}
                            pickle.dump(pearson_to_pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
                # print(f'{nlp_model}-layer{layer}-seq{seq_len}, mean_pearson: {mean_pearson_across_subjects}, {pearson_dict}')

                # Export to CSV and chart visualizations
                df_row = {'expt_setting': expt_setting, 'nlp_model': nlp_model, 'model_type': get_base_model_type(nlp_model), 'base_or_booksum': get_base_or_booksum(nlp_model),
                          'layer': layer, 'seq_len': seq_len, 
                          'mean_pearson': mean_pearson_across_subjects, **pearson_dict}
                df_row = pd.DataFrame([df_row])
                df_out = pd.concat([df_out, df_row], ignore_index=True)

# === Save result to disk ===
out_dir = '0-logs/pearson'
os.makedirs(out_dir, exist_ok=True)

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_pearson_scores.csv')
df_out.to_csv(out_csv_path, index=False)

# # === Generate figures and save to PNG ===
# df_viz = df_out[['nlp_model', 'layer', 'seq_len', 'mean_pearson']]
df_viz = df_out

# FIGURE - X-axis: aggregate pearson score, Y-axis: Discourse feature, single bar plot
out_fig_path = os.path.join(out_dir, 'plot_discourse_features_barplot.png')
plt = sns.barplot(x = "mean_pearson", y = "expt_setting", data = df_out,
                  hue = "base_or_booksum",
                  orient = "h",
                  ci = None)    # Remove error bars
plt.set(xlim=(0.0, 0.032))

plt.set_ylabel('Discourse feature')
plt.set_xlabel('Pearson correlation')
for container in plt.containers:
    plt.bar_label(container, fmt='%.3f', label_type='edge')
    # plt.bar_label(container, fmt='%.3f', label_type='edge', padding = -30)
fig = plt.figure
fig.subplots_adjust(top=0.9)
fig.suptitle('Mean Pearson correlation for different discourse features', fontsize=10)
fig.savefig(out_fig_path, bbox_inches='tight')

# # X-axis: Sequence length, Col: nlp_model
# out_fig_path = os.path.join(out_dir, 'plot_seq_len_row_layer_col_nlp_model.png')
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "expt_setting", style = "expt_setting", kind = "line", markers=True,
#                     row = 'layer',
#                     col = "nlp_model")
# fig = plt.figure
# fig.savefig(out_fig_path)

# # X-axis: Sequence length, Col: nlp_model
# out_fig_path = os.path.join(out_dir, 'plot_seq_len_col_nlp_model.png')
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "expt_setting", style = "expt_setting", kind = "line", markers=True,
#                     col = "nlp_model")
# fig = plt.figure
# fig.savefig(out_fig_path)

# # X-axis: Sequence length, Row: expt_setting, Col: layer
# out_fig_path = os.path.join(out_dir, 'plot_seq_len_row_expt_setting_col_layer.png')
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "nlp_model", style = "nlp_model", kind = "line", markers=True,
#                     row = "expt_setting",
#                     col = "layer")
# fig = plt.figure
# fig.savefig(out_fig_path)

# # X-axis: Sequence length, Row: expt_setting, Col: model_type
# out_fig_path = os.path.join(out_dir, 'plot_seq_len_row_expt_setting_col_model_type.png')
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "nlp_model", style = "nlp_model", kind = "line", markers=True,
#                     row = "expt_setting",
#                     col = "model_type")
# fig = plt.figure
# fig.savefig(out_fig_path)

# # X-axis: Sequence length, Row: model_type, Col: layer
# out_fig_path = os.path.join(out_dir, 'plot_seq_len_row_model_type_col_expt_setting.png')
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "nlp_model", style = "nlp_model", kind = "line", markers=True,
#                     row = "model_type",
#                     col = "expt_setting")
# fig = plt.figure
# fig.savefig(out_fig_path)

# # X-axis: Layer, Row: nlp_model, Col: Sequence length
# out_fig_path = os.path.join(out_dir, 'plot_layer_row_nlp_model_col_seq_len.png')
# plt = sns.relplot(x = "layer", y = "mean_pearson", data = df_viz, 
#                     hue = "expt_setting", style = "expt_setting", kind = "line", markers=True,
#                     row = 'nlp_model',
#                     col = "seq_len")
# fig = plt.figure
# fig.savefig(out_fig_path)

# # X-axis: Layer, Row: model_type, Col: nlp_model
# out_fig_path = os.path.join(out_dir, 'plot_seq_len_row_base_or_booksum_col_model_type.png')
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "expt_setting", style = "expt_setting", kind = "line", markers=True,
#                     row = 'base_or_booksum',
#                     col = "model_type")
# fig = plt.figure
# fig.savefig(out_fig_path)


# # Play around to generate the desired here
# out_fig_path = '0-logs/pearson/PLAY_AROUND.png'
# plt = sns.relplot(x = "seq_len", y = "mean_pearson", data = df_viz, 
#                     hue = "nlp_model", style = "nlp_model", kind = "line", markers=True,
#                     row = 'model_type',
#                     col = "expt_setting")
# fig = plt.figure
# fig.savefig(out_fig_path)