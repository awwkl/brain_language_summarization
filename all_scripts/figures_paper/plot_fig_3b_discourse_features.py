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
legend_title = 'LongT5'
nlp_model_list = []
# nlp_model_list += ['bart-base', 'bart-booksum']
# nlp_model_list += ['led-base', 'led-booksum']
# nlp_model_list += ['bigbird-base', 'bigbird-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']
# nlp_model_list += ['bart-booksum', 'led-booksum', 'bigbird-booksum', 'long-t5-booksum']

# expt_setting_list = ['Full']
expt_setting_list = ['Characters', 'Emotion', 'Motion']
# expt_setting_list +=  ['Non-discourse', 'Full']

nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

seq_len_list = [20, 100, 200, 500, 700, 1000]
# layer_list = [6, 7, 8, 9, 10, 11, 12]
layer_list = range(1,13)
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

df_out = pd.DataFrame()
df_expanded = pd.DataFrame()
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
                
                # Export expanded version to CSV
                for subj in subject_list:
                    df_expanded_row = {'expt_setting': expt_setting, 'nlp_model': nlp_model, 'model_type': get_base_model_type(nlp_model), 'base_or_booksum': get_base_or_booksum(nlp_model),
                          'layer': layer, 'seq_len': seq_len, 
                          'subject': subj, 'subj_pearson': pearson_dict[subj]}
                    df_expanded_row = pd.DataFrame([df_expanded_row])
                    df_expanded = pd.concat([df_expanded, df_expanded_row], ignore_index=True)

# === Save result to disk ===
out_dir = '0-logs/figures_paper/fig_3b/'
os.makedirs(out_dir, exist_ok=True)

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_pearson_scores.csv')
df_out.to_csv(out_csv_path, index=False)

df_expanded.to_csv(os.path.join(out_dir, 'df_pearson_scores_expanded.csv'), index=False)

# # === Generate figures and save to PNG ===
# df_viz = df_out[['nlp_model', 'layer', 'seq_len', 'mean_pearson']]
df_viz = df_out
# from matplotlib import pyplot
# pyplot.figure(figsize=(15, 15))

# Default values
padding_amt = 20
xlim_start = 0
xlim_end = 0.032
if nlp_model_list[0].replace('-base', '') == 'bart':
    padding_amt = 22
    xlim_start = -0.01
    xlim_end = 0.032
elif nlp_model_list[0].replace('-base', '') == 'led':
    padding_amt = 30
    xlim_start = -0.020
elif nlp_model_list[0].replace('-base', '') == 'bigbird':
    xlim_start = -0.01
    xlim_end = 0.036
elif nlp_model_list[0].replace('-base', '') == 'long-t5':
    xlim_end = 0.046
    

# FIGURE - X-axis: aggregate pearson score, Y-axis: Discourse feature, single bar plot
out_fig_path = os.path.join(out_dir, 'plot_discourse_features_barplot.png')
plt = sns.barplot(x = "mean_pearson", y = "expt_setting", data = df_out,
                  hue = "base_or_booksum",
                  orient = "h",
                  errorbar='se')
plt.set(xlim=(xlim_start, xlim_end))

plt.set_ylabel('')
# plt.set_ylabel(nlp_model_list)
plt.set_xlabel('Pearson correlation', fontsize=15)
plt.legend(title='', fontsize=15)
plt.tick_params(axis='both', labelsize=15)
# sns.move_legend(plt, "upper left", bbox_to_anchor=(0.7,0.2)) 
sns.move_legend(plt, "upper left", bbox_to_anchor=(0.60,0.25), fontsize=15) 


for container in plt.containers:
    plt.bar_label(container, fmt='%.3f', label_type='edge', padding=padding_amt, fontsize=13)
    # plt.bar_label(container, fmt='%.3f', label_type='edge', padding = -30)
fig = plt.figure
fig.subplots_adjust(top=0.93)
# fig.suptitle('Pearson correlation for various discourse features', fontsize=15)
fig.suptitle(legend_title, fontsize=15)
fig.savefig(out_fig_path, bbox_inches='tight')
fig.savefig(out_fig_path.replace('.png', '.svg'), bbox_inches='tight')
