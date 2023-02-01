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
# nlp_model_list += ['bart-base', 'bart-booksum']
# nlp_model_list += ['led-base', 'led-booksum']
nlp_model_list += ['bigbird-base', 'bigbird-booksum']
# nlp_model_list += ['long-t5-base', 'long-t5-booksum']
legend_title = 'BigBird'

nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

seq_len_list = [20, 100, 200, 300, 400, 500, 700, 1000]
# seq_len_list = [20, 100, 200, 300, 400, 500]
# layer_list = range(0,33)
# layer_list = [1, 2, 4, 6, 8, 10, 11, 12]
layer_list = [2, 6, 10, 14, 18, 24, 28, 32]
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

df_out = pd.DataFrame()
for nlp_model in nlp_model_list:
    base_or_booksum = 'booksum' if '-booksum' in nlp_model else 'base'
    for layer in layer_list:
        for seq_len in seq_len_list:
            
            acc_dict = {}
            for subject in subject_list:
                pkl_path = f'3-eval-results/{nlp_model}/predict_{subject}_with_{nlp_model}_layer_{layer}_len_{seq_len}_accs.pkl'
                if not os.path.isfile(pkl_path):
                    continue
                
                acc = extract_brain_score(pkl_path)
                acc_dict[subject] = acc

            # If no data was found, just skip and do not add
            if len(acc_dict) < len(subject_list):
                print(f'@@@ Incomplete: {nlp_model}-layer{layer}-seq{seq_len}, # subjects = {len(acc_dict)} ===')
                continue

            # Compute mean accuracy, and round all values to 1 decimal place
            brain_score_across_subjects = 100.0 * sum(acc_dict.values()) / len(acc_dict)
            brain_score_across_subjects = round(brain_score_across_subjects, 1)
            for subj in acc_dict.keys():
                acc_dict[subj] = round(100.0 * acc_dict[subj], 1)
            print(f'{nlp_model}-layer{layer}-seq{seq_len}, brain_score: {brain_score_across_subjects:.1f}, #: {len(acc_dict)}, {acc_dict}')
                
            # Export to CSV and chart visualizations
            for subj, acc in acc_dict.items():
                df_row = {'nlp_model': nlp_model, 'base_or_booksum': base_or_booksum, 'layer': layer, 'seq_len': seq_len,
                          'subject': subj, 'brain_score': acc}
                df_row = pd.DataFrame([df_row])
                df_out = pd.concat([df_out, df_row], ignore_index=True)

# === Save results to disk ===
out_dir = '0-logs/figures_paper/fig_1a/'
os.makedirs(out_dir, exist_ok=True)

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_brain_score.csv')
df_out.to_csv(out_csv_path, index=False)

# Generate figures as PNG
df_viz = df_out
sns.set(font_scale=1.8)
# sns.set_style('darkgrid', {'legend.frameon':True})

# X-axis: Sequence length, 1 graph for each layer
out_fig_path = os.path.join(out_dir, 'plot_seq_len_col_layer.png')
plt = sns.relplot(x = "seq_len", y = "brain_score", data = df_viz, 
                    hue = "base_or_booksum", style = "base_or_booksum", col = "layer", kind = "line", markers=True,
                    dashes = ['', (1,1)], linewidth=5, markersize=12,
                    col_wrap=4,
                    err_style='bars',
                    errorbar='se')
plt.set_ylabels('20vs20 accuracy')
plt.set_xlabels('Sequence length')

# bbox_anchor_coords = (0.83, 0.29)
bbox_anchor_coords = (0.83, 0.48)
sns.move_legend(plt, "upper left", bbox_to_anchor=bbox_anchor_coords, title=legend_title, 
                frameon=True, facecolor='white', title_fontsize = 'medium', fontsize='medium', markerscale=2, numpoints=1,
                handlelength=3.5)
# plt._legend.set_title(legend_title)

for legobj in plt._legend.legendHandles:
    legobj.set_linewidth(5.0)

# handles, lables = plt.get_legend_handles_labels()
# for h in handles:
#     h.set_markersize(500)

# for lh in plt._legend.legendHandles: 
#     # lh.set_alpha(1)
#     lh._sizes = [500] 

fig = plt.figure
fig.tight_layout()
fig.savefig(out_fig_path, bbox_inches='tight')
fig.savefig(out_fig_path.replace('.png', '.svg'), bbox_inches='tight')
