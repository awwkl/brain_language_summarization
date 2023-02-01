import torch
import numpy as np
from transformers import AutoTokenizer, LongT5EncoderModel
import time as tm

# Use GPU if possible
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_total_layers = 12 # total number of layers in model

@torch.inference_mode()
def get_long_t5_layer_representations(args, text_array, remove_chars, word_ind_to_extract):
    seq_len = args.sequence_length
    nlp_model = args.nlp_model

    if nlp_model == 'long-t5-base':
        hf_name ='google/long-t5-tglobal-base'
    elif nlp_model == 'long-t5-booksum':
        hf_name ='pszemraj/long-t5-tglobal-base-16384-book-summary'

    model = LongT5EncoderModel.from_pretrained(hf_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model.eval()

    # get the token embeddings
    token_embeddings = []
    for word in text_array:
        current_token_embedding = get_long_t5_token_embeddings([word], tokenizer, model, remove_chars)
        token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))
    
    # where to store layer-wise long_t5 embeddings of particular length
    LongT5 = {}
    for layer in range(n_total_layers):
        LongT5[layer] = []
    LongT5[-1] = token_embeddings

    # Before we've seen enough words to make up the seq_len
    # Extract index 0 after supplying tokens 0 to 0, extract 1 after 0 to 1, 2 after 0 to 2, ... , 19 after 0 to 19
    start_time = tm.time()
    for truncated_seq_len in range(1, 1+seq_len):
        word_seq = text_array[:truncated_seq_len]
        from_start_word_ind_to_extract = -1 + truncated_seq_len
        LongT5 = add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model, remove_chars, 
                                                            from_start_word_ind_to_extract, LongT5)
        if truncated_seq_len % 100 == 0:
            print('Completed {} out of {}: {}'.format(truncated_seq_len, len(text_array), tm.time()-start_time))
            start_time = tm.time()

    word_seq = text_array[:seq_len]
    if word_ind_to_extract < 0: # the index is specified from the end of the array, so invert the index
        from_start_word_ind_to_extract = seq_len + word_ind_to_extract
    else:
        from_start_word_ind_to_extract = word_ind_to_extract
        
    # Then, use sequences of length seq_len, still adding the embedding of the last word in a sequence
    for end_curr_seq in range(seq_len, len(text_array)):
        word_seq = text_array[end_curr_seq-seq_len+1:end_curr_seq+1]
        LongT5 = add_avrg_token_embedding_for_specific_word(word_seq, tokenizer, model, remove_chars,
                                                            from_start_word_ind_to_extract, LongT5)

        if end_curr_seq % 100 == 0:
            print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time()-start_time))
            start_time = tm.time()

    print('Done extracting sequences of length {}'.format(seq_len))
    return LongT5

# extracts layer representations for all words in words_in_array
# encoded_layers: list of tensors, length num layers. each tensor of dims num tokens by num dimensions in representation
# word_ind_to_token_ind: dict that maps from index in words_in_array to index in array of tokens when words_in_array is tokenized,
#                       with keys: index of word, and values: array of indices of corresponding tokens when word is tokenized
@torch.inference_mode()
def predict_long_t5_embeddings(words_in_array, tokenizer, model, remove_chars):    
    for word in words_in_array:
        if word in remove_chars:
            print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1
    
    n_seq_tokens = 0
    seq_tokens = []
    
    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    
    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)
            
        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1
    
    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    # Use local attention, do not use global attention
    # attention_mask = torch.ones(tokens_tensor.shape, dtype=torch.long, device=tokens_tensor.device)
    # global_attention_mask = torch.zeros(tokens_tensor.shape, dtype=torch.long, device=tokens_tensor.device)

    outputs = model(input_ids=tokens_tensor, output_hidden_states=True)
    hidden_states = outputs['hidden_states'][1:]
    
    return hidden_states, word_ind_to_token_ind, None
  
# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
@torch.inference_mode()
def add_word_long_t5_embedding(model_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        model_dict[specific_layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.cpu().detach().numpy()
            model_dict[layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0)) # avrg over all tokens for specified word
    return model_dict

# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# tokenizer: Long-T5 tokenizer
# model: Long-T5 model
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# model_dict: where to save the extracted embeddings
@torch.inference_mode()
def add_avrg_token_embedding_for_specific_word(word_seq,tokenizer,model,remove_chars,from_start_word_ind_to_extract,model_dict):
    
    word_seq = list(word_seq)
    all_sequence_embeddings, word_ind_to_token_ind, _ = predict_long_t5_embeddings(word_seq, tokenizer, model, remove_chars)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_long_t5_embedding(model_dict, all_sequence_embeddings,token_inds_to_avrg)
    
    return model_dict


# get the Long-T5 token embeddings
@torch.inference_mode()
def get_long_t5_token_embeddings(words_in_array, tokenizer, model, remove_chars):    
    for word in words_in_array:
        if word in remove_chars:
            print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1
    
    n_seq_tokens = 0
    seq_tokens = []
    
    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    
    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word
        word_tokens = tokenizer.tokenize(word)
            
        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1
    
    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    
    # outputs = model(tokens_tensor, output_hidden_states=True)
    # hidden_states = outputs['encoder_hidden_states']
    # token_embeddings = hidden_states[0].cpu()
    
    input_embedding_module = model.base_model.get_input_embeddings()
    token_embeddings = input_embedding_module(tokens_tensor).cpu()
    
    return token_embeddings
