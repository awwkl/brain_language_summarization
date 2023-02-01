import os
import pickle
import pandas as pd
import seaborn as sns

def extract_brain_score(pkl_path):
    loaded = pickle.load(open(pkl_path, 'rb'))          # (4, ~27905)
    mean_subj_acc_across_folds = loaded.mean(0)
    mean_overall = mean_subj_acc_across_folds.mean(0)

    return mean_overall

def extract_rouge_results(pkl_path):
    raw_dict = pickle.load(open(pkl_path, 'rb'))

    rouge1_mid = raw_dict['rouge1'][1]          # 1 for mid from (low, mid, high)
    rouge2_mid = raw_dict['rouge2'][1]
    rougeL_mid = raw_dict['rougeL'][1]
    rougeLsum_mid = raw_dict['rougeLsum'][1]

    rouge1_mid_fmeasure = rouge1_mid[2]         # 2 for fmeasure from (precision, recall, fmeasure)
    rouge2_mid_fmeasure = rouge2_mid[2]
    rougeL_mid_fmeasure = rougeL_mid[2]
    rougeLsum_mid_fmeasure = rougeLsum_mid[2]

    result_dict = {'rouge1': rouge1_mid_fmeasure, 'rouge2': rouge2_mid_fmeasure, 'rougeL': rougeL_mid_fmeasure, 'rougeLsum': rougeLsum_mid_fmeasure}
    for k,v in result_dict.items():
        result_dict[k] = round(100*v, 2)

    return result_dict

# === Change these variables ===
# Change NLP models to analyze
nlp_model_list = []
# nlp_model_list += ['led-base', 'bart-base']
nlp_model_list += ['led-base']
nlp_model_list += ['bart-base']

# Change which set of experiments to analyze
expt_name = 'my_finetuned'
epochs_list = range(1,21)
# checkpoint_num_list = [600*e for e in epochs_list]
checkpoint_num_list = [600, 1200, 1800, 2400, 3000, 6000]

layer_list = [11]
# seq_len_list = [500]
seq_len_list = [20, 100, 200, 500]
subject_list = ['F', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

# === Ensure unique entries and sorted alphabetically ===
nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

df_out = pd.DataFrame()
for nlp_model in nlp_model_list:
    # Add ROUGE results
    print('=== Add ROUGE results ===')
    for checkpoint_num in checkpoint_num_list:
        pkl_path = f'6-rouge-score/{expt_name}/{nlp_model}/checkpoint-{checkpoint_num}/rouge_results.pkl'
        if not os.path.isfile(pkl_path):
            continue
        
        result_dict = extract_rouge_results(pkl_path)
        print(f'{nlp_model}/checkpoint-{checkpoint_num}', result_dict)

        # Export to CSV and chart visualizations
        for rouge_type, rouge_score in result_dict.items():
            df_row = {'nlp_model': nlp_model, 'checkpoint': checkpoint_num, 'score_type': rouge_type, 'score': rouge_score}
            df_row = pd.DataFrame([df_row])
            df_out = pd.concat([df_out, df_row], ignore_index=True)

    # Add eval loss results
    print('=== Add eval_loss results ===')
    eval_loss_pkl_path = f'0-logs/rouge/eval_loss_for_{nlp_model}.pkl'
    eval_loss_dict = pickle.load(open(eval_loss_pkl_path, 'rb'))
    for checkpoint_num, eval_loss in eval_loss_dict.items():
        df_row = {'nlp_model': nlp_model, 'checkpoint': checkpoint_num, 'score_type': 'eval_loss', 'score': eval_loss}
        df_row = pd.DataFrame([df_row])
        df_out = pd.concat([df_out, df_row], ignore_index=True)
        
    # Add brain score results
    print('=== Add brain score results ===')
    model_type = nlp_model.replace('-base', '')
    for checkpoint_num in checkpoint_num_list:
        model_name_with_checkpoint = f'{model_type}-checkpoint-{checkpoint_num}'
        overall_brain_score_list = []
        
        for layer in layer_list:
            for seq_len in seq_len_list:
                for subject in subject_list:
                    pkl_path = f'3-eval-results/{model_name_with_checkpoint}/predict_{subject}_with_{model_name_with_checkpoint}_layer_{layer}_len_{seq_len}_accs.pkl'
                    if not os.path.isfile(pkl_path):
                        print('not a file:', pkl_path)
                        continue
                    acc = extract_brain_score(pkl_path)
                    overall_brain_score_list.append(acc)
        
        # if len(overall_brain_score_list) < ( len(layer_list)*len(seq_len_list)*len(subject_list)):
        if len(overall_brain_score_list) < ( len(layer_list)*len(seq_len_list)*len(subject_list)):
            print(f'brain scores not enough for {model_name_with_checkpoint}:', len(overall_brain_score_list))
            continue
        
        print(f'Added brain score for {model_name_with_checkpoint}')
        overall_brain_score = sum(overall_brain_score_list) / len(overall_brain_score_list)
        df_row = {'nlp_model': nlp_model, 'checkpoint': checkpoint_num, 'score_type': 'brain_score', 'score': overall_brain_score}
        df_row = pd.DataFrame([df_row])
        df_out = pd.concat([df_out, df_row], ignore_index=True)

# Save raw results to CSV
os.makedirs('0-logs/rouge/', exist_ok=True)
out_csv_path = '0-logs/rouge/df_rouge_scores.csv'
df_out.to_csv(out_csv_path, index=False)

# === Generate figures and save to PNG ===
df_viz = df_out
drop_score_types = ['rouge2', 'rougeL', 'rougeLsum']
drop_row_indices = df_viz['score_type'].isin(drop_score_types)
df_viz = df_viz.drop(df_viz.index[ drop_row_indices ])

# X-axis: Checkpoint #, 1 graph for each ROUGE type
n_col_wrap = 6 - len(drop_score_types)
out_fig_path = '0-logs/rouge/plot_checkpoint_col_rouge.png'
plt = sns.relplot(x = "checkpoint", y = "score", data = df_viz, 
                    hue = "nlp_model", style = "nlp_model", col = "score_type", kind = "line", markers=True,
                    col_wrap=n_col_wrap, 
                    facet_kws={'sharey': False, 'sharex': True})
fig = plt.figure
fig.savefig(out_fig_path)
