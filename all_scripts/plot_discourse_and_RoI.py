import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns

np.random.seed(42)  # Set random seed

# roi_label options can be found in loaded.item()[subject].keys()
# Shapes for 1 subject: {'all': (5018,), 'PostTemp': (1234,), 'AntTemp': (783,), 'AngularG': (380,), 
#                        'IFG': (379,), 'MFG': (289,), 'IFGorb': (252,), 'pCingulate': (957,), 'dmpfc': (744,)}
roi_label_list = ['all', 'PostTemp', 'AntTemp', 'AngularG', 'IFG', 'MFG', 'IFGorb', 'pCingulate', 'dmpfc', 'Non-language']
HP_subj_roi_inds_loaded = np.load('./data/HP_subj_roi_inds.npy', allow_pickle=True)

def extract_pearson_per_roi(all_voxels_pearson, subject, roi_label):            # (~27905,)
    # RoI mask for voxels outside language region is simply the logical NOT of the region 'all'
    if roi_label == 'Non-language':
        region_all_mask = HP_subj_roi_inds_loaded.item()[subject]['all']
        roi_mask = np.logical_not(region_all_mask)
    else:
        roi_mask = HP_subj_roi_inds_loaded.item()[subject][roi_label]           # (~27905,) => [True,False,False,True,...]

    brain_score_voxels_roi = all_voxels_pearson[roi_mask]                       # (~783,)
    return brain_score_voxels_roi.mean()                                        # scalar

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
nlp_model_list += ['bart-base', 'bart-booksum']
nlp_model_list += ['led-base', 'led-booksum']
nlp_model_list += ['bigbird-base', 'bigbird-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']

# seq_len_list = [20]
seq_len_list = [20, 100, 200, 500]
seq_len_list += [700, 1000]
layer_list = [6, 7, 8, 9, 10, 11, 12]
# layer_list = [6]
# layer_list = [1, 2, 4]
# layer_list = [11, 12]
# layer_list = [1, 2, 4, 6, 8, 10, 11, 12]
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
# subject_list = ['F', 'H']

discourse_feature_list = ['Characters', 'Emotion', 'Motion', 'Non-discourse']

# === Automatically set ===
nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

df_out = pd.DataFrame()
for nlp_model in nlp_model_list:
    base_model_name = nlp_model.replace('-booksum', '').replace('-base', '')
    base_or_booksum = 'booksum' if 'booksum' in nlp_model else 'base'
    os.makedirs(f'8-RoI-and-pearson-saved/{nlp_model}', exist_ok=True)
    
    for layer in layer_list:
        for seq_len in seq_len_list:
            n_subjects = 0
            for subject in subject_list:
                # If Pearson has been computed for ALL discourse features and ALL RoI labels, just load it
                pearson_saved_path = f'8-RoI-and-pearson-saved/{nlp_model}/{nlp_model}_layer_{layer}_len_{seq_len}_subject_{subject}.pkl'
                if os.path.isfile(pearson_saved_path):
                    pearson_discourse_roi_dict = pickle.load(open(pearson_saved_path, 'rb'))
                
                # Otherwise, compute the whole dictionary, then save it to a pkl file for future loading
                else:
                    pearson_discourse_roi_dict = {}
                    
                    pred_path = f'2-encoding_predictions/{nlp_model}/predict_{subject}_with_{nlp_model}_layer_{layer}_len_{seq_len}.npy'
                    if not os.path.isfile(pred_path):
                        continue

                    for discourse_feature in discourse_feature_list:
                        pearson_discourse_dict = {}
                        
                        all_voxels_pearson = extract_all_voxels_pearson(pred_path, discourse_feature)
                        for roi_label in roi_label_list:
                            pearson_discourse_dict[roi_label] = extract_pearson_per_roi(all_voxels_pearson, subject, roi_label)

                        pearson_discourse_roi_dict[discourse_feature] = pearson_discourse_dict
                        
                    # Save the final dict for ALL discourse features and ALL RoI labels
                    with open(pearson_saved_path, 'wb') as handle:
                        pickle.dump(pearson_discourse_roi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                n_subjects += 1
                for discourse_feature in discourse_feature_list:
                    for roi_label in roi_label_list:
                        df_row = {'nlp_model': nlp_model, 'layer': layer, 'seq_len': seq_len, 'subject': subject,
                                'base_model_name': base_model_name, 'base_or_booksum': base_or_booksum,
                                'discourse_feature': discourse_feature, 'roi_label': roi_label, 
                                'mean_pearson': pearson_discourse_roi_dict[discourse_feature][roi_label]}
                        df_row = pd.DataFrame([df_row])
                        df_out = pd.concat([df_out, df_row], ignore_index=True)

            if n_subjects < len(subject_list):
                print(f'@@@ Incomplete: {nlp_model}_layer_{layer}_len_{seq_len}, # subjects: {n_subjects}')
            else:
                print(f'Completed all subjects: {nlp_model}_layer_{layer}_len_{seq_len}, # subjects: {n_subjects}')

# === Save result to disk ===
out_dir = '0-logs/discourse_and_RoI'
os.makedirs(out_dir, exist_ok=True)

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_RoI_and_pearson.csv')
df_out.to_csv(out_csv_path, index=False)

# # === Generate figures and save to PNG ===
df_viz = df_out
df_orig = df_out

base_model_name_list = [nlp_model.replace('-booksum', '').replace('-base', '') for nlp_model in nlp_model_list]
base_model_name_list = list( set(base_model_name_list) )

layer_list = [str(x) for x in layer_list]
seq_len_list = [str(x) for x in seq_len_list]

layer_list_string = ', '.join(layer_list)
seq_len_list_string = ', '.join(seq_len_list)
subject_list_string = ', '.join(subject_list)

# FIGURE - Col: Discourse feature, horizontal barplot, Y-axis: RoI label, X-axis: mean_pearson
out_fig_path = os.path.join(out_dir, 'plot_RoI_col_discourse.png')

base_or_booksum_types = df_viz['base_or_booksum'].unique()

# plt = sns.FacetGrid(data = df_viz, col = 'discourse_feature')
# plt.map(sns.barplot, 'mean_pearson', 'roi_label', 
#                         order = 'base_or_booksum_types',    # Very important so they are not drawn on top of each other
#                         hue = "base_or_booksum",
#                         orient = "h", ci = None)

plt = sns.catplot(data = df_viz, x = 'mean_pearson', y = 'roi_label', orient = 'h', kind = 'bar',
                  col = 'discourse_feature', hue = 'base_or_booksum', 
                  ci = None)

fig = plt.figure
from textwrap import wrap
plt_title = f'models: {base_model_name_list}, layers: [{layer_list_string}], sequence lengths: [{seq_len_list_string}], subjects: [{subject_list_string}]'
# fig.suptitle("\n".join(wrap(plt_title)), fontsize=8)
fig.subplots_adjust(top=0.9)
fig.suptitle(plt_title)
fig.savefig(out_fig_path, bbox_inches='tight')
import matplotlib
matplotlib.pyplot.clf()

# FIGURE - In a single plot, X-axis: RoI label, Y-axis: mean pearson, Hue: Discourse feature
df_base = df_orig.loc[df_orig['base_or_booksum'] == 'base'].drop(columns=['layer', 'seq_len'])                # Get rows for base, and also booksum models
df_booksum = df_orig.loc[df_orig['base_or_booksum'] == 'booksum'].drop(columns=['layer', 'seq_len'])

df_base = df_base.groupby(['discourse_feature', 'roi_label']).mean().reset_index()                          # 4 discourse features * 10 roi labels = 40 rows output, each with a mean value for pearson
df_booksum = df_booksum.groupby(['discourse_feature', 'roi_label']).mean().reset_index()

assert df_booksum[['discourse_feature', 'roi_label']].equals(df_base[['discourse_feature', 'roi_label']])   # Ensure that the rows for base and booksum dataframes are correctly aligned
df_booksum_minus_base = df_booksum.copy()[['discourse_feature', 'roi_label']]
df_booksum_minus_base['pearson_difference'] = df_booksum['mean_pearson'] - df_base['mean_pearson']          # Compute booksum - base, for mean pearson, grouped by ['discourse_feature', 'roi_label']

out_csv_path = os.path.join(out_dir, 'df_RoI_and_pearson_booksum_minus_base.csv')
df_booksum_minus_base.to_csv(out_csv_path, index=False)

out_fig_path = os.path.join(out_dir, 'plot_RoI_and_pearson_booksum_minus_base.png')
plt = sns.barplot(x = 'roi_label', y = 'pearson_difference', data = df_booksum_minus_base,
                  hue = 'discourse_feature',
                  order = roi_label_list)
fig = plt.figure
# fig.suptitle("\n".join(wrap(plt_title)), fontsize=8)
fig.suptitle('difference of: booksum - base\n' + plt_title)
fig.savefig(out_fig_path, bbox_inches='tight')
matplotlib.pyplot.clf()

# FIGURE - Same as above, but with error bars (I think this procedure is not fair because the sampling population is not homogenous)
df_base = df_orig.loc[df_orig['base_or_booksum'] == 'base']
df_booksum = df_orig.loc[df_orig['base_or_booksum'] == 'booksum']

group_by_columns = ['base_model_name', 'layer', 'seq_len', 'subject', 'discourse_feature', 'roi_label']
df_base = df_base.groupby(group_by_columns).mean().reset_index()
df_booksum = df_booksum.groupby(group_by_columns).mean().reset_index()

assert df_booksum[group_by_columns].equals(df_base[group_by_columns])
df_booksum_minus_base = df_booksum.copy()[group_by_columns]
df_booksum_minus_base['pearson_difference'] = df_booksum['mean_pearson'] - df_base['mean_pearson']

out_csv_path = os.path.join(out_dir, 'df_RoI_and_pearson_booksum_minus_base_FULL.csv')
df_booksum_minus_base.to_csv(out_csv_path, index=False)

out_fig_path = os.path.join(out_dir, 'plot_RoI_and_pearson_booksum_minus_base_FULL.png')
plt = sns.barplot(x = 'roi_label', y = 'pearson_difference', data = df_booksum_minus_base,
                  hue = 'discourse_feature',
                  order = roi_label_list)
fig = plt.figure
# fig.suptitle("\n".join(wrap(plt_title)), fontsize=8)
fig.suptitle('difference of: booksum - base\n' + plt_title)
fig.savefig(out_fig_path, bbox_inches='tight')