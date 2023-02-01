# Training language models for deeper understanding improves brain alignment

This repository contains code for the paper "Training language models for deeper understanding improves brain alignment"

## Setup
### Install packages
- `pip install -r requirements.txt`

### Data - fMRI Recordings of 8 Subjects Reading Harry Potter
- Download the already [preprocessed data here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8?usp=sharing). This data contains fMRI recordings for 8 subjects reading one chapter of Harry Potter. The data been detrended, smoothed, and trimmed to remove the first 20TRs and the last 15TRs. For more information about the data, refer to the paper. We have also provided the precomputed voxel neighborhoods that we have used to compute the searchlight classification accuracies. 
- Place it under the data folder in this repository (e.g. `./data/fMRI/` and `./data/voxel_neighborhoods`).

## Running Code
Below, we provide instructions for how to run the various experiments and code that we use in our paper. 
- The sections below are ordered to match the flow of the paper as closely as possible.
- In our paper, we run a large number of experiments (many models, layers, sequence lengths, subjects, discourse features, brain ROIs, etc). Hence, we provide scripts to automate the process of running experiments across the various models, layers, etc. Hopefully, this will make it as easy as possible for others to use our code efficiently.

### 1. Extract NLP representations
- Change the variables in this main script and run it: `all_scripts/scripts/outer_nlp_extract.sh`
- The main script calls this inner script: `all_scripts/scripts/inner_nlp_extract.sh`
- The inner script calls this python file: `extract_nlp_features.py`
- The output will be generated in: `1-nlp_features/`

### 2. Align NLP representations to human brain activity (i.e. linear encoding process)
- Change the variables in this main script and run it: `all_scripts/scripts/outer_encoding.sh`
- The main script calls this inner script: `all_scripts/scripts/inner_encoding.sh`
- The inner script calls this python file: `predict_brain_from_nlp.py`
- The output will be generated in: `2-encoding_predictions/`

### 3. Evaluate brain-NLP alignment using 20v20 classification accuracy
- The script described in the section above also performs the 20v20 evaluation: `all_scripts/scripts/outer_encoding.sh`
- The main script calls this inner script: `all_scripts/scripts/inner_encoding.sh`
- The inner script calls this python file: `evaluate_brain_predictions.py`
- The output will be generated in: `3-eval-results/`

### 4. Compute Language modeling ability (i.e. perplexity or cross-entropy loss)
- Change the variables in this main script and run it: `all_scripts/scripts/outer_perplexity.sh`
- The main script calls this inner script: `all_scripts/scripts/inner_perplexity.sh`
- The inner script calls this python file: `calculate_perplexity.py`
- The output will be generated in: `4-perplexity-results/`

### 5. Train language models on BookSum (not used for paper)
- Change the variables in this main script and run it: `all_scripts/scripts/outer_finetune_booksum.sh`
- The main script calls this python file: `finetune_booksum.py`
- The output will be generated in: `5-finetune-booksum/`

### 6. Compute performance of models on BookSum dataset (i.e. ROUGE score) (not used for paper)
- Change the variables in this main script and run it: `all_scripts/scripts/outer_rouge.sh`
- The main script calls this python file: `compute_rouge.py`
- The output will be generated in: `6-rouge-score/`

### 7. Interpretability method to compute Pearson correlation for various discourse features
- First, we need to label the words in the Harry Potter text with their discourse features
    - Download it from: [http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/](http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/)
    - Place it at `data/story_features.mat`
- Next, run `align_story_feature_TRs.ipynb` to map the labeled words to fMRI TRs
- Finally, extract the TRs corresponding to each discourse feature, and compute the Pearson correlation score for the discourse feature
    - Use the python file: `all_scripts/plot_pearson.py`
    - The output will be generated in: `7-pearson-saved/`

### 8. Compute Pearson correlation for each pair of (discourse feature, brain ROI) (not used for paper)
- Use the python file: `all_scripts/plot_discourse_and_RoI.py`
- The output will be generated in: `8-RoI-and-pearson-saved/`

### 9. Generate brain voxel values for visualizing on brain plots
- Change the variables in this main script and run it: `all_scripts/scripts/outer_generate_voxel_values.sh`
- The main script calls this python file: `all_scripts/generate_voxel_values.py`
- The output will be generated in: `9-pearson-voxels-for-brain-plot/`

### Others
- To plot the key figures used for paper, see: `all_scripts/figures_paper/`
- To plot other figures for visualization, see: `all_scripts/plot_{}`
- To run significance tests and false discovery rate (FDR) correction using the Benjamini–Hochberg (BH) procedure, see: `compute_stat_significance.ipynb`
- Our repository uses code from the following [GitHub repository](https://github.com/mtoneva/brain_language_nlp)
