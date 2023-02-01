import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns

# roi_label options can be found in loaded.item()[subject].keys()
# Shapes for 1 subject: {'all': (5018,), 'PostTemp': (1234,), 'AntTemp': (783,), 'AngularG': (380,), 
#                        'IFG': (379,), 'MFG': (289,), 'IFGorb': (252,), 'pCingulate': (957,), 'dmpfc': (744,)}
roi_label_list = ['all', 'PostTemp', 'AntTemp', 'AngularG', 'IFG', 'MFG', 'IFGorb', 'pCingulate', 'dmpfc', 'Non-language']

def extract_brain_scores_per_roi(brain_score_pkl_path, subject):
    brain_score_loaded = pickle.load(open(brain_score_pkl_path, 'rb'))          # (4, ~27905)
    brain_score_voxels_ALL = brain_score_loaded.mean(0)                         # (~27905,)
    HP_subj_roi_inds_loaded = np.load('./data/HP_subj_roi_inds.npy', allow_pickle=True)

    brain_scores_per_roi = np.empty([len(roi_label_list)])                      # will be: [58.83, 59.21, ...]
    for ind, roi_label in enumerate(roi_label_list):
        # RoI mask for voxels outside language region is simply the logical NOT of the region 'all'
        if roi_label == 'Non-language':
            region_all_mask = HP_subj_roi_inds_loaded.item()[subject]['all']
            roi_mask = np.logical_not(region_all_mask)
        else:
            roi_mask = HP_subj_roi_inds_loaded.item()[subject][roi_label]       # (~27905,) => [True,False,False,True,...]

        brain_score_voxels_roi = brain_score_voxels_ALL[roi_mask]               # (~783,)
        brain_score_voxels_roi_mean = brain_score_voxels_roi.mean()             # scalar
        brain_scores_per_roi[ind] = round(100*brain_score_voxels_roi_mean, 2)
    return brain_scores_per_roi                                                 # (10,)

# === Change these variables ===
legend_title = 'LongT5'
nlp_model_list = []
# nlp_model_list += ['led-base', 'led-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']
# nlp_model_list += ['bigbird-base', 'bigbird-booksum']
# nlp_model_list += ['bart-base', 'bart-booksum']

seq_len_list = [20, 100, 200, 300, 400, 500]
seq_len_list += [700, 1000]
# layer_list = [6, 8, 10, 11, 12]
# layer_list = [1, 2, 4]
# layer_list = [11, 12]
layer_list = [1, 2, 4, 6, 8, 10, 11, 12]
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

# seq_len_list = [20]
# layer_list = [6]
# subject_list = ['F']


# === Automatically set ===
nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

df_out = pd.DataFrame()
for nlp_model in nlp_model_list:
    base_model_name = nlp_model.replace('-booksum', '').replace('-base', '')
    base_or_booksum = 'booksum' if 'booksum' in nlp_model else 'base'
    
    for layer in layer_list:
        for seq_len in seq_len_list:
            
            brain_scores_per_roi_multiple_subjects = []
            for subject in subject_list:
                pkl_path = f'3-eval-results/{nlp_model}/predict_{subject}_with_{nlp_model}_layer_{layer}_len_{seq_len}_accs.pkl'
                if not os.path.isfile(pkl_path):
                    continue
                
                brain_scores_per_roi = extract_brain_scores_per_roi(pkl_path, subject)
                brain_scores_per_roi_multiple_subjects.append(brain_scores_per_roi)

            # If # of subjects is less than the subjects we want, data is not complete, do NOT export
            if len(brain_scores_per_roi_multiple_subjects) < len(subject_list):
                print(f'@@@ Incomplete: {nlp_model}-layer{layer}-seq{seq_len}, # subjects = {len(brain_scores_per_roi_multiple_subjects)}')
                continue
            
            brain_scores_per_roi_multiple_subjects = np.stack(brain_scores_per_roi_multiple_subjects, axis=0)   # (8 subj, 9 rois)
            brain_scores_per_roi_averaged_across_subjects = brain_scores_per_roi_multiple_subjects.mean(0)      # (9 rois,)

            # Export to CSV and chart visualizations
            for ind, roi_label in enumerate(roi_label_list):
                df_row = {'nlp_model': nlp_model, 'layer': layer, 'seq_len': seq_len, 
                          'base_model_name': base_model_name, 'base_or_booksum': base_or_booksum,
                          'roi_label': roi_label, 'roi_brain_score': brain_scores_per_roi_averaged_across_subjects[ind]}
                df_row = pd.DataFrame([df_row])
                df_out = pd.concat([df_out, df_row], ignore_index=True)

# === Save result to disk ===
out_dir = '0-logs/figures_paper/fig_2b/'
os.makedirs(out_dir, exist_ok=True)

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_RoI_brain_score.csv')
df_out.to_csv(out_csv_path, index=False)

# === Generate figures and save to PNG ===
df_viz = df_out

base_model_name_list = [nlp_model.replace('-booksum', '').replace('-base', '') for nlp_model in nlp_model_list]
base_model_name_list = list( set(base_model_name_list) )

layer_list = [str(x) for x in layer_list]
seq_len_list = [str(x) for x in seq_len_list]

layer_list_string = ', '.join(layer_list)
seq_len_list_string = ', '.join(seq_len_list)
subject_list_string = ', '.join(subject_list)

# FIGURE - Get complete aggregate plot
out_fig_path = os.path.join(out_dir, 'plot_average_compare_RoI.png')
plt = sns.barplot(x = "roi_brain_score", y = "roi_label", data = df_viz,
                  hue = "base_or_booksum",
                  errorbar='se',
                  orient = "h")
plt.set(xlim=(52, 70))

plt.set_xlabel('20vs20 accuracy', fontsize=13)
plt.set_ylabel(nlp_model_list)
# plt.set_ylabel('')
plt.legend(title='Model type')
plt.tick_params(axis='both', labelsize=13)
plt.legend(fontsize=13)

fig = plt.figure
from textwrap import wrap
# plt_title = f'{base_model_name_list}, layers: [{layer_list_string}], sequence lengths: [{seq_len_list_string}], subjects: [{subject_list_string}]'
# plt.set_title("\n".join(wrap(plt_title, width=100)), fontsize=8)
# plt.legend(title='@@@ Test @@@')
# plt.set_title('20vs20 Brain score for various brain language Regions of Interest (ROI)                           ', fontsize=13)
plt.set_title(legend_title, fontsize=13)
fig.subplots_adjust(top=0.9)
fig.savefig(out_fig_path, bbox_inches='tight')
fig.savefig(out_fig_path.replace('.png', '.svg'), bbox_inches='tight')




# FIGURE - X-axis: Sequence length, 1 graph for each RoI
out_fig_path = os.path.join(out_dir, 'plot_seq_len_col_RoI.png')
plt = sns.relplot(x = "seq_len", y = "roi_brain_score", data = df_viz, 
                    hue = "nlp_model", style = "nlp_model", col = "roi_label", kind = "line", markers=True,
                    col_wrap=5)

fig = plt.figure
fig.subplots_adjust(top=0.9)
plt_title = f'layers: [{layer_list_string}], sequence lengths: [{seq_len_list_string}], subjects: [{subject_list_string}]'
fig.suptitle(plt_title)
fig.savefig(out_fig_path)
fig.savefig(out_fig_path.replace('.png', '.svg'), bbox_inches='tight')
