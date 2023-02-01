from utils.models.bert_utils import get_bert_layer_representations
from utils.models.longformer_utils import get_longformer_layer_representations
from utils.models.led_utils import get_led_layer_representations
from utils.models.long_t5_utils import get_long_t5_layer_representations
from utils.models.bigbird_utils import get_bigbird_layer_representations
from utils.models.bart_utils import get_bart_layer_representations
import time as tm
import numpy as np
import torch
import os
import argparse

def save_layer_representations(model_layer_dict, model_name, seq_len, save_dir):             
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_{}_layer_{}.npy'.format(save_dir,model_name,seq_len,layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

model_options =  ['bert', 'longformer']
model_options += ['led-base', 'led-booksum']
model_options += ['long-t5-base', 'long-t5-booksum']
model_options += ['bigbird-base', 'bigbird-booksum']
model_options += ['bart-base', 'bart-booksum']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default=None)
    parser.add_argument("--sequence_length", type=int, default=1, help='length of context to provide to NLP model (default: 1)')
    parser.add_argument("--output_dir", required=True, help='directory to save extracted representations to')
    parser.add_argument("--checkpoint_dir", default=None, help='directory from which to load model checkpoint')

    args = parser.parse_args()
    print(args)
    
    text_array = np.load(os.getcwd() + '/data/stimuli_words.npy')
    remove_chars = [",","\"","@"]
    
    
    if args.nlp_model == 'bert':
        # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
        # for CLS, set to 0; for SEP set to -1; for last word set to -2
        word_ind_to_extract = -2
        nlp_features = get_bert_layer_representations(args, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model == 'longformer':
        word_ind_to_extract = -1
        nlp_features = get_longformer_layer_representations(args, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model.startswith('led-'):
        word_ind_to_extract = -1
        nlp_features = get_led_layer_representations(args, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model.startswith('long-t5-'):
        word_ind_to_extract = -1
        nlp_features = get_long_t5_layer_representations(args, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model.startswith('bigbird-'):
        word_ind_to_extract = -1
        nlp_features = get_bigbird_layer_representations(args, text_array, remove_chars, word_ind_to_extract)
    elif args.nlp_model.startswith('bart-'):
        word_ind_to_extract = -1
        nlp_features = get_bart_layer_representations(args, text_array, remove_chars, word_ind_to_extract)
    else:
        print('Unrecognized model name {}'.format(args.nlp_model))
        
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)          
              
    save_layer_representations(nlp_features, args.nlp_model, args.sequence_length, args.output_dir)
        
        
        
        
    
    
    

    
