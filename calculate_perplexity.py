import os
import argparse
import pickle
from utils.perplexity import calc_multi_fold_perplexity 

model_options =  ['bert', 'longformer']
model_options += ['led-base', 'led-booksum']
model_options += ['long-t5-base', 'long-t5-booksum']
model_options += ['bigbird-base', 'bigbird-booksum']
model_options += ['bart-base', 'bart-booksum']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='bert', choices=model_options)
    parser.add_argument("--output_dir", required=True, help='directory to save perplexity results to')
    
    args = parser.parse_args()
    print(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate perplexity for multiple CV folds
    ret_perplexity_list, trained_weights = calc_multi_fold_perplexity(args.nlp_model, args.output_dir)
    
    pkl_dict = {'perplexity_list': ret_perplexity_list, 'trained_weights': trained_weights}
    perplexity_pkl_path = os.path.join(args.output_dir, 'perplexity.pkl')
    with open(perplexity_pkl_path, 'wb') as handle:
        pickle.dump(pkl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
