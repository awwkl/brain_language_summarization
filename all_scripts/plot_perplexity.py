import os
import numpy as np
import pickle as pk
import pandas as pd
import seaborn as sns

def extract_results(pkl_path):
    loaded = pk.load(open(pkl_path, 'rb'))
    trained_weights = loaded['trained_weights']
    perplexity_list = loaded['perplexity_list']
    loss_list = [ np.log(x) for x in perplexity_list ]

    decimal_places = 2
    mean_perplexity = round( sum(perplexity_list) / len(perplexity_list), decimal_places)
    mean_loss = round( sum(loss_list) / len(loss_list), decimal_places)
    perplexity_list = [ round(p,decimal_places) for p in perplexity_list ]
    loss_list = [ round(loss,decimal_places) for loss in loss_list ]

    return trained_weights, mean_perplexity, perplexity_list, mean_loss, loss_list

# === Change these variables ===
# Change NLP models to analyze
nlp_model_list = []
nlp_model_list += ['bert', 'longformer']
nlp_model_list += ['led-base', 'led-booksum']
nlp_model_list += ['long-t5-base', 'long-t5-booksum']
nlp_model_list += ['bigbird-base', 'bigbird-booksum']
nlp_model_list += ['bart-base', 'bart-booksum']
nlp_model_list += ['led-booksum', 'long-t5-booksum', 'bigbird-booksum', 'bart-booksum']

# Change which set of experiments to analyze
expt_name = 'freeze_early5'

# === Ensure unique entries and sorted alphabetically ===
nlp_model_list = list( set(nlp_model_list) )
nlp_model_list.sort()

df_out = pd.DataFrame()
df_averaged = pd.DataFrame()
for nlp_model in nlp_model_list:
    pkl_path = f'4-perplexity-results/{expt_name}/{nlp_model}/perplexity.pkl'
    if not os.path.isfile(pkl_path):
        continue
    
    trained_weights, mean_perplexity, perplexity_list, mean_loss, loss_list = extract_results(pkl_path)
    trained_weights = f'{len(trained_weights)} weights' if len(trained_weights) > 5 else trained_weights
    # print(f'{nlp_model}: perplexity = {mean_perplexity}, trained = {trained_weights}, list = {perplexity_list}')
    print(f'{nlp_model}: loss = {mean_loss}, trained = {trained_weights}, list = {loss_list}')
    
    base_model_name = nlp_model.replace('-booksum', '').replace('-base', '')
    base_or_booksum = 'booksum' if 'booksum' in nlp_model else 'base'

    for cv_loss in loss_list:
        df_row = {'nlp_model': nlp_model, 'expt_name': expt_name, 
                    'mean_perplexity': mean_perplexity, 'mean_loss': mean_loss, 'cv_loss': cv_loss,
                    'base_model_name': base_model_name, 'base_or_booksum': base_or_booksum}
        df_row = pd.DataFrame([df_row])
        df_out = pd.concat([df_out, df_row], ignore_index=True)

    df_averaged_row = {'nlp_model': nlp_model, 'mean_loss': mean_loss,
              'cv_loss_0': loss_list[0], 'cv_loss_1': loss_list[1], 'cv_loss_2': loss_list[2], 'cv_loss_3': loss_list[3]}
    df_averaged_row = pd.DataFrame([df_averaged_row])
    df_averaged = pd.concat([df_averaged, df_averaged_row], ignore_index=True)


# === Save results to disk ===
out_dir = '0-logs/perplexity'

# Save to CSV
out_csv_path = os.path.join(out_dir, 'df_perplexity_raw.csv')
df_out.to_csv(out_csv_path, index=False)

# Save averaged results to CSV
out_csv_path = '0-logs/perplexity/df_perplexity_averaged.csv'
df_averaged.to_csv(out_csv_path, index=False)

# === Generate figures and save to PNG ===
# Generate figure, split by layer
# out_fig_path = '0-logs/perplexity/plot_perplexity_col_base_model_name.png'

# col_order = ['led', 'long-t5', 'bigbird', 'bart']
# col_order.sort()
# plt = sns.FacetGrid(df_out, col="base_model_name", col_order=col_order)
# plt.map(sns.barplot, 'base_or_booksum', 'cv_loss', order=['base', 'booksum'])

# fig = plt.figure
# fig.subplots_adjust(top=0.8)
# fig.suptitle(expt_name)
# fig.savefig(out_fig_path) # use `bbox_inches='tight'` if needed to prevent X-labels from being cut off
# del plt, fig

# Plot all the models together
out_fig_path = os.path.join(out_dir, 'plot_perplexity_together.png')
plt = sns.barplot(x = "cv_loss", y = "base_model_name", data = df_out,
                  hue = "base_or_booksum",
                  orient = "h")
# plt.set(xlim=(52,72))

plt.set_xlabel('Language modeling cross-entropy loss for 4 folds')
fig = plt.figure
fig.subplots_adjust(top=0.9)
fig.suptitle(expt_name)
fig.savefig(out_fig_path, bbox_inches='tight')
