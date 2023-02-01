import torch
import numpy as np
import os
import pickle

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def compare_old_versus_new_bert():
    # Compare new and old npy files to make sure outputs are the same
    old_npy_dir = 'nlp_features/old_bert/'
    new_npy_dir = 'nlp_features/'

    old_npy_files = [f for f in os.listdir(old_npy_dir) if f.endswith('.npy')]
    old_npy_files.sort()

    max_diff_list = []
    max_percent_diff_list = []
    for old_file_name in old_npy_files:
        # Get file path for old and new numpy object
        old_file_path = os.path.join(old_npy_dir, old_file_name)
        new_file_path = os.path.join(new_npy_dir, old_file_name)

        # Load old and new numpy object
        old_npy_obj = np.load(old_file_path)
        new_npy_obj = np.load(new_file_path)
        assert old_npy_obj.shape == new_npy_obj.shape

        # Compute absolute difference between old and new numpy object
        diff = np.abs(old_npy_obj - new_npy_obj)
        max_diff = np.max(diff)

        # Compute (percent) absolute difference between old and new numpy object
        percent_diff = 100 * diff / np.abs(old_npy_obj)
        max_percent_diff = np.max(percent_diff)

        # Append the results to relevant lists
        max_diff_list.append(max_diff)
        max_percent_diff_list.append(max_percent_diff)

    print(max_diff_list)
    print('max_diff', max(max_diff_list))

    print(max_percent_diff_list)
    print('max_percent_diff', max(max_percent_diff_list))

def inspect_fMRI_data():
    fMRI_dir = 'data/fMRI/'
    fMRI_files = [f for f in os.listdir(fMRI_dir)]

    # (1211, 24983), (1211, 27905), ..., (1211, 24678)
    fMRI_subject_files = [f for f in fMRI_files if f.startswith('data_subject')]
    for f in fMRI_subject_files:
        data = np.load( os.path.join(fMRI_dir, f) )
        print(data.shape)

def get_model_output(words_in_array, tokenizer, model):
    seq_tokens = []
    for word in words_in_array:
        word_tokens = tokenizer.tokenize(word)
        for token in word_tokens:
            seq_tokens.append(token)
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    outputs = model(tokens_tensor, output_hidden_states=True)
    encoder_hidden_states = outputs['encoder_hidden_states'][1:]    # This is a tuple: (layer1, layer2, ..., layer6)
    decoder_hidden_states = outputs['decoder_hidden_states'][1:]
    all_layers_hidden_states = encoder_hidden_states + decoder_hidden_states    # Join tuples
    
    return all_layers_hidden_states

# This method checks that output is the same for both
# AutoModelForSeq2SeqLM and LEDModel
def compare_led_models():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import LEDTokenizer, LEDModel

    hf_name ='pszemraj/led-base-book-summary'
    HP_text_array = np.load(os.getcwd() + '/data/stimuli_words.npy')
    HP_text_array = HP_text_array[:100]

    LM_model = AutoModelForSeq2SeqLM.from_pretrained(hf_name).to(device)
    LM_tokenizer = AutoTokenizer.from_pretrained(hf_name)
    
    BASE_model = LEDModel.from_pretrained(hf_name).to(device)
    BASE_tokenizer = LEDTokenizer.from_pretrained(hf_name)

    LM_model_hidden_states = get_model_output(HP_text_array, LM_tokenizer, LM_model)
    BASE_model_hidden_states = get_model_output(HP_text_array, BASE_tokenizer, BASE_model)

    for layer_num in range(len(LM_model_hidden_states)):
        LM_layer = LM_model_hidden_states[layer_num]
        BASE_layer = BASE_model_hidden_states[layer_num]
        print( f'layer {layer_num}: {torch.equal(LM_layer, BASE_layer)}' )


def test_long_t5():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    hf_name ='google/long-t5-tglobal-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
    outputs = model(inputs, output_hidden_states=True)

    hidden_states = outputs['hidden_states'][1:]
    print(hidden_states)

def test_bigbird():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    hf_name ='pszemraj/bigbird-pegasus-large-K-booksum'
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
    outputs = model(inputs, output_hidden_states=True)

    encoder_hidden_states = outputs['encoder_hidden_states'][1:]    # This is a tuple: (layer1, layer2, ..., layer6)
    decoder_hidden_states = outputs['decoder_hidden_states'][1:]
    all_layers_hidden_states = encoder_hidden_states + decoder_hidden_states    # Join tuples
    print()

def test_bart():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    hf_name = 'KamilAin/bart-base-booksum'
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
    outputs = model(inputs, output_hidden_states=True)

    encoder_hidden_states = outputs['encoder_hidden_states'][1:]    # This is a tuple: (layer1, layer2, ..., layer6)
    decoder_hidden_states = outputs['decoder_hidden_states'][1:]
    all_layers_hidden_states = encoder_hidden_states + decoder_hidden_states    # Join tuples
    print()
    
def compare_truncated_features_old_new():
    seq_len = 20
    layers = [6, 8, 10]
    for layer in layers:
        old_features = np.load(f'./1-nlp_features/old_bart-booksum/bart-booksum_length_{seq_len}_layer_{layer}.npy')
        new_features = np.load(f'./1-nlp_features/bart-booksum/bart-booksum_length_{seq_len}_layer_{layer}.npy')
        
        old_features_end = old_features[seq_len:]
        new_features_end = new_features[seq_len:]
        print('Equal at end:', np.array_equal(old_features_end, new_features_end))
        
        old_features_start = old_features[:seq_len]
        new_features_start = new_features[:seq_len]
        print('Equal at start:', np.array_equal(old_features_start, new_features_start))

def get_nlp_to_hf_name_dict():
    nlp_to_hf_name_dict = {
        'led-base':         'allenai/led-base-16384',
        'led-booksum':      'pszemraj/led-base-book-summary',
        'long-t5-base':     'google/long-t5-tglobal-base',
        'long-t5-booksum':  'pszemraj/long-t5-tglobal-base-16384-book-summary',
        'bigbird-base':     'google/bigbird-pegasus-large-bigpatent',
        'bigbird-booksum':  'pszemraj/bigbird-pegasus-large-K-booksum',
        'bart-base':        'facebook/bart-base',
        'bart-booksum':     'KamilAin/bart-base-booksum'
    }
    return nlp_to_hf_name_dict

def get_number_of_tokens_per_word_HP_data():
    from transformers import AutoTokenizer

    HP_text_array = np.load(os.getcwd() + '/data/stimuli_words.npy')
    output_dict_nlp_model_to_num_tokens_list = {}

    nlp_to_hf_name_dict = get_nlp_to_hf_name_dict()
    for nlp_model, hf_name in nlp_to_hf_name_dict.items():
        num_tokens_per_word_list = []
        tokenizer = AutoTokenizer.from_pretrained(hf_name)

        for word in HP_text_array:
            word_tokens = tokenizer.tokenize(word)
            num_tokens = len(word_tokens)
            num_tokens_per_word_list.append(num_tokens)
            
        output_dict_nlp_model_to_num_tokens_list[nlp_model] = num_tokens_per_word_list
    
    pkl_path = os.path.join('0-logs/', 'num_tokens_for_each_model.pkl')
    with open(pkl_path, 'wb') as handle:
        pickle.dump(output_dict_nlp_model_to_num_tokens_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def explore_number_of_tokens_per_word_HP_data():
    pkl_path = os.path.join('0-logs/', 'num_tokens_for_each_model.pkl')
    output_dict_nlp_model_to_num_tokens_list = pickle.load(open(pkl_path, 'rb'))
    
    seq_len_list = [20, 100, 200, 500, 600, ]
    nlp_to_hf_name_dict = get_nlp_to_hf_name_dict()
    for nlp_model, hf_name in nlp_to_hf_name_dict.items():
        num_tokens_per_word_list = output_dict_nlp_model_to_num_tokens_list[nlp_model]
        n_words = len(num_tokens_per_word_list)

        print('Average # tokens per word for {nlp_model}:', sum(num_tokens_per_word_list) / n_words)
        
        for seq_len in seq_len_list:
            total_tokens_list = []
            for end_word_ind in range(0, n_words):
                start_word_ind = max(0, end_word_ind - seq_len + 1)
                sub_list = num_tokens_per_word_list[start_word_ind:1+end_word_ind]
                total_tokens = sum(sub_list)
                total_tokens_list.append(total_tokens)
            print(f'Max # tokens for {nlp_model}, seq_len {seq_len}:', max(total_tokens_list))
        
def test_led_from_checkpoint():
    from transformers import LEDConfig, LEDModel, AutoTokenizer, AutoModelForSeq2SeqLM
    
    nlp_model = 'led-checkpoint'
    checkpoint_dir = '5-finetune-booksum/my_finetuned/led-base/checkpoint-6000'

    if nlp_model == 'led-checkpoint':
        model = LEDModel.from_pretrained(checkpoint_dir)
        model.config.max_decoder_position_embeddings = 16384
        tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384')
    else:
        if nlp_model == 'led-base':
            hf_name = 'allenai/led-base-16384'
        elif nlp_model == 'led-booksum':
            hf_name = 'pszemraj/led-base-book-summary'

        config = LEDConfig.from_pretrained(hf_name)
        config.max_decoder_position_embeddings = 16384      # Process tokens longer than 1024
        model = LEDModel(config)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    print()

# def rename_nlp_feature_files():
#     features_dir = '1-nlp_features/bart-checkpoint-3000'
#     os.chdir(features_dir)
#     print(os.getcwd())
#     # os.mknod('bart-checkpoint_test')
#     for f in os.listdir(os.getcwd()):
#         if 'bart-checkpoint_' in f:
#             old_name = f
#             new_name = f.replace('bart-checkpoint_', 'bart-checkpoint-3000_')
#             print('old_name:', old_name)
#             print('new_name:', new_name)
#             # os.rename(old_name, new_name)
#             print()

def get_and_save_eval_loss_finetune_booksum():
    nlp_model = 'led-base'
    import json, pickle
    path = f'5-finetune-booksum/my_finetuned/{nlp_model}/checkpoint-12000/trainer_state.json'
    f = open(path)
    data = json.load(f)
    output_dict = {}
    for epoch_dict in data['log_history']:
        if 'eval_loss' in epoch_dict:
            print(epoch_dict)
            output_dict[ epoch_dict['step'] ] = epoch_dict['eval_loss']
    print(output_dict)
    
    pkl_path = os.path.join('0-logs/rouge', f'eval_loss_for_{nlp_model}.pkl')
    with open(pkl_path, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def test_word_to_TR():
    data = np.load('./data/fMRI/data_subject_{}.npy'.format('F')) # (1211, ~27905)
    n_words = data.shape[0]
    ind = CV_ind(n_words, n_folds=4)

    ind_num = 0
    train_ind = ind!=ind_num

    TR_train_indicator = train_ind
    SKIP_WORDS=20
    END_WORDS=5176

    time = np.load('./data/fMRI/time_fmri.npy')                 # (1351,)
    runs = np.load('./data/fMRI/runs_fmri.npy')                 # (1351,)
    time_words = np.load('./data/fMRI/time_words_fmri.npy')     # (5176,)
    time_words = time_words[SKIP_WORDS:END_WORDS]               # (5156,)
        
    word_train_indicator = np.zeros([len(time_words)], dtype=bool)      # (5156,)
    words_id = np.zeros([len(time_words)],dtype=int)                    # (5156,)
    # w=find what TR each word belongs to
    for i in range(len(time_words)):                
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
        if words_id[i] <= len(runs) - 15:
            offset = runs[int(words_id[i])]*20 + (runs[int(words_id[i])]-1)*15
            if TR_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True

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

    result_dict = {'rouge-1': rouge1_mid_fmeasure, 'rouge-2': rouge2_mid_fmeasure, 'rouge-L': rougeL_mid_fmeasure, 'rouge-Lsum': rougeLsum_mid_fmeasure}
    for k,v in result_dict.items():
        result_dict[k] = round(100*v, 3)

    return result_dict

def extract_rouge_for_bart():
    results = extract_rouge_results('6-rouge-score/HF_booksum/bart-booksum/checkpoint-0/rouge_results.pkl')
    print(results)

# === Functions that can be run ===
# inspect_fMRI_data()
# compare_old_versus_new_bert()
# compare_led_models()
# test_long_t5()
# test_bigbird()
# test_bart()
# compare_truncated_features_old_new()
# get_number_of_tokens_per_word_HP_data()
# explore_number_of_tokens_per_word_HP_data()
# test_led_from_checkpoint()
# rename_nlp_feature_files()
# get_and_save_eval_loss_finetune_booksum()
# test_word_to_TR()
extract_rouge_for_bart()