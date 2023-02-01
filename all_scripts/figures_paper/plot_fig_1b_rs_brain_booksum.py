import os
import pickle
import pandas as pd
import seaborn as sns

def extract_brain_score(pkl_path):
    loaded = pickle.load(open(pkl_path, 'rb'))          # (4, ~27905)
    mean_subj_acc_across_folds = loaded.mean(0)
    mean_overall = mean_subj_acc_across_folds.mean(0)

    return mean_overall

rouge_dict_for_models = {
    # Reference: https://paperswithcode.com/sota/summarization-on-kmfoda-booksum
    'long-t5-booksum':  {'rouge-1': 36.408, 'rouge-2': 6.065, 'rouge-L': 16.721, 'rouge-Lsum': 33.34},
    'bigbird-booksum':  {'rouge-1': 34.085, 'rouge-2': 5.922, 'rouge-L': 16.389	, 'rouge-Lsum': 31.616},
    'led-booksum':      {'rouge-1': 33.454, 'rouge-2': 5.223, 'rouge-L': 16.204, 'rouge-Lsum': 29.977},

    # Reference: Computed ROUGE scores using `compute_rouge.py`
    'bart-booksum':     {'rouge-1': 28.91, 'rouge-2': 6.315, 'rouge-L': 13.83, 'rouge-Lsum': 13.819},
}
    

# === Change these variables ===
nlp_model_list = []
nlp_model_list += ['led-booksum', 'bigbird-booksum', 'long-t5-booksum', 'bart-booksum']
# nlp_model_list += ['led-booksum', 'bigbird-booksum', 'long-t5-booksum']

seq_len_list = [20, 100, 200, 300, 400, 500]
# layer_list = [1, 2, 4, 6, 8, 10, 11, 12]
layer_list = [6, 8, 10, 11, 12]

# === No need to change these ===
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
nlp_model_list = list( set(nlp_model_list) )
# nlp_model_list.sort()

df_out = pd.DataFrame()
for nlp_model in nlp_model_list:

    # 1. Get Brain score
    nlp_model_brain_score_list = []  # Single score for the NLP model
    for layer in layer_list:
        for seq_len in seq_len_list:
            
            acc_dict = {}
            for subject in subject_list:
                pkl_path = f'3-eval-results/{nlp_model}/predict_{subject}_with_{nlp_model}_layer_{layer}_len_{seq_len}_accs.pkl'
                if not os.path.isfile(pkl_path):
                    continue
                
                acc = extract_brain_score(pkl_path)
                acc_dict[subject] = acc

            # If no data was found, just skip
            if len(acc_dict) == 0:
                continue

            # Compute mean accuracy, and round all values to 1 decimal place
            mean_acc_across_subjects = 100.0 * sum(acc_dict.values()) / len(acc_dict)
            for subj in acc_dict.keys():
                acc_dict[subj] = round(100.0 * acc_dict[subj], 1)
                
            # If # of subjects is less than the subjects we want, data is not complete, do NOT export
            if len(acc_dict) < len(subject_list):
                print(f'=== Data still running, # subjects = {len(acc_dict)} ===')
                continue
            nlp_model_brain_score_list.append(mean_acc_across_subjects)

    if len(nlp_model_brain_score_list) < ( len(layer_list)*len(seq_len_list) ):
        print(f'=== Data incomplete for {nlp_model}: {len(nlp_model_brain_score_list)} ===')
        continue
    nlp_model_brain_score = round(sum(nlp_model_brain_score_list) / len(nlp_model_brain_score_list), 2)
    
    # 2. Get ROUGE scores in a dictionary, e.g. {'rouge-1': 36.408, 'rouge-2': 6.065, ...}
    rouge_dict = rouge_dict_for_models[nlp_model]

    # Export to CSV and chart visualizations
    for rouge_score_type, rouge_score in rouge_dict.items():
        df_row = {'nlp_model': nlp_model, 'brain_score': nlp_model_brain_score, 'rouge_score_type': rouge_score_type, 'rouge_score': rouge_score}
        df_row = pd.DataFrame([df_row])
        df_out = pd.concat([df_out, df_row], ignore_index=True)
    
    
# === Save results to disk ===
out_dir = '0-logs/figures_paper/fig_1b/'
os.makedirs(out_dir, exist_ok=True)

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_rs_brain_booksum.csv')
df_out.to_csv(out_csv_path, index=False)

# # === Generate figures and save to PNG ===
sns.set(font_scale=1.4)
df_viz = df_out

out_fig_path = os.path.join(out_dir, 'plot_brain_vs_ALL_rouge.png')
plt = sns.relplot(x = "brain_score", y = "rouge_score", data = df_viz, kind = "scatter", s=500,
                    hue = "nlp_model", style = "nlp_model", 
                    hue_order = nlp_model_list, style_order = nlp_model_list,
                    col = "rouge_score_type",
                    facet_kws={'sharey': False, 'sharex': True})
fig = plt.figure
fig.savefig(out_fig_path)
fig.savefig(out_fig_path.replace('.png', '.svg'))
