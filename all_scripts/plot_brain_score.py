import os
import pickle
import pandas as pd
import seaborn as sns

def extract_brain_score(pkl_path):
    loaded = pickle.load(open(pkl_path, 'rb'))          # (4, ~27905)
    mean_subj_acc_across_folds = loaded.mean(0)         # (~27905,)
    mean_overall = mean_subj_acc_across_folds.mean(0)   # scalar

    return mean_overall

nlp_model_list = []
# nlp_model_list += ['bert', 'longformer']
# nlp_model_list += ['led-base', 'led-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']
# nlp_model_list += ['bigbird-base', 'bigbird-booksum']
# nlp_model_list += ['bart-base', 'bart-booksum']
# nlp_model_list += ['led-booksum', 'long-t5-booksum', 'bigbird-booksum', 'bart-booksum']
# nlp_model_list += ['bart-base', 'bart-booksum', 'bart-checkpoint-3000']
# nlp_model_list += ['bart-checkpoint-6000', 'bart-checkpoint-9000', 'bart-checkpoint-12000']
# nlp_model_list += ['led-base', 'led-booksum', 'led-checkpoint-9000']
# nlp_model_list += ['led-base', 'led-booksum', 'led-checkpoint-9000']

nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

seq_len_list = [20, 100, 200, 300, 400, 500]
seq_len_list += [700, 1000]
# layer_list = [4, 6, 8, 10, 11, 12]
layer_list = list(range(13))
# subject_list = ['F', 'H']
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

df_out = pd.DataFrame()
for nlp_model in nlp_model_list:
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
            mean_acc_across_subjects = round(mean_acc_across_subjects, 1)
            for subj in acc_dict.keys():
                acc_dict[subj] = round(100.0 * acc_dict[subj], 1)
            print(f'{nlp_model}-layer{layer}-seq{seq_len}, mean_acc: {mean_acc_across_subjects:.1f}, #: {len(acc_dict)}, {acc_dict}')
                
            # If # of subjects is less than the subjects we want, data is not complete, do NOT export
            if len(acc_dict) < len(subject_list):
                print(f'=== Data still running, # subjects = {len(acc_dict)} ===')
                continue
            
            # Export to CSV and chart visualizations
            df_row = {'nlp_model': nlp_model, 'layer': layer, 'seq_len': seq_len, 'mean_acc': mean_acc_across_subjects, **acc_dict}
            df_row = pd.DataFrame([df_row])
            df_out = pd.concat([df_out, df_row], ignore_index=True)

# Save to CSV
out_csv_path = '0-logs/brain_score/df_brain_score.csv'
df_out.to_csv(out_csv_path, index=False)

# === Generate figures and save to PNG ===
df_viz = df_out[['nlp_model', 'layer', 'seq_len', 'mean_acc']]

# X-axis: Sequence length, 1 graph for each layer
out_fig_path = '0-logs/brain_score/plot_seq_len_col_layer.png'
plt = sns.relplot(x = "seq_len", y = "mean_acc", data = df_viz, 
                    hue = "nlp_model", style = "nlp_model", col = "layer", kind = "line", markers=True,
                    col_wrap=4)
fig = plt.figure
fig.savefig(out_fig_path)

n_col_wrap = len(nlp_model_list) if len(nlp_model_list) <= 3 else 4
# X-axis: Sequence length, 1 graph for each NLP model
out_fig_path = '0-logs/brain_score/plot_seq_len_col_nlp_model.png'
plt = sns.relplot(x = "seq_len", y = "mean_acc", data = df_viz, 
                    hue = "layer", style = "layer", col = "nlp_model", kind = "line", markers=True,
                    col_wrap=n_col_wrap)
fig = plt.figure
fig.savefig(out_fig_path)

# X-axis: Layer, 1 graph for each NLP model
out_fig_path = '0-logs/brain_score/plot_layer_col_nlp_model.png'
plt = sns.relplot(x = "layer", y = "mean_acc", data = df_viz, 
                    hue = "seq_len", style = "seq_len", col = "nlp_model", kind = "line", markers=True,
                    col_wrap=n_col_wrap)
fig = plt.figure
fig.savefig(out_fig_path)

# X-axis: Layer, 1 graph for each Sequence length
out_fig_path = '0-logs/brain_score/plot_layer_col_seq_len.png'
plt = sns.relplot(x = "layer", y = "mean_acc", data = df_viz, 
                    hue = "nlp_model", style = "nlp_model", col = "seq_len", kind = "line", markers=True,
                    col_wrap=3)
fig = plt.figure
fig.savefig(out_fig_path)
