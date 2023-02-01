import os
import argparse
from utils.booksum import finetune_model_on_booksum 

model_options = []
model_options += ['led-base', 'led-booksum']
model_options += ['long-t5-base', 'long-t5-booksum']
model_options += ['bigbird-base', 'bigbird-booksum']
model_options += ['bart-base', 'bart-booksum']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='bert', choices=model_options)
    parser.add_argument("--output_dir", required=True, help='directory to save finetuned models and logs')
    
    args = parser.parse_args()
    print(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Finetune model on booksum
    finetune_model_on_booksum(args.nlp_model, args.output_dir)
    