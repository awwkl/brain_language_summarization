import os
import argparse
import pickle
from utils.booksum import compute_rouge 
from utils.perplexity import get_hf_name

model_options = []
model_options += ['led-base', 'led-booksum']
model_options += ['long-t5-base', 'long-t5-booksum']
model_options += ['bigbird-base', 'bigbird-booksum']
model_options += ['bart-base', 'bart-booksum']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='bert', choices=model_options)
    parser.add_argument("--checkpoint_dir", required=True, help='directory from which to load model checkpoint')
    parser.add_argument("--output_dir", required=True, help='directory to save rouge results')
    args = parser.parse_args()

    # If model is booksum model, load checkpoint from HF repositories instead of local directory
    if '-booksum' in args.nlp_model:
        args.checkpoint_dir = get_hf_name(args.nlp_model)
        
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compute rouge scores
    rouge_results = compute_rouge(args.nlp_model, args.checkpoint_dir)

    # Save rouge results
    rouge_results_pkl_path = os.path.join(args.output_dir, 'rouge_results.pkl')
    with open(rouge_results_pkl_path, 'wb') as handle:
        pickle.dump(rouge_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
