import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import dataset_dict
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_hf_name(nlp_model):
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
    return nlp_to_hf_name_dict[nlp_model] # Intended to throw exception if nlp_model not in dict

def get_HP_text_array():
    text_array = np.load('./data/stimuli_words.npy')
    return text_array

def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        ind[i*n_items:(i+1)*n_items] = i
    ind[(n_folds-1)*n_items:] = (n_folds-1)
    return ind

def group_texts(examples):
    block_size = 128
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def create_LM_dataset_dict(train_words, val_words, test_words, tokenizer):
    # Create Dataset and DatasetDict objects
    train_dict = {'text': train_words}
    train_dataset = Dataset.from_dict(train_dict)
    val_dict = {'text': val_words}
    val_dataset = Dataset.from_dict(val_dict)
    test_dict = {'text': test_words}
    test_dataset = Dataset.from_dict(test_dict)
    lm_dataset_dict = dataset_dict.DatasetDict({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})

    # Tokenize dataset to input_ids
    lm_dataset_dict = lm_dataset_dict.map(lambda example: tokenizer([" ".join(x) for x in example["text"]]), 
                                            batched=True, num_proc=4, remove_columns=lm_dataset_dict['train'].column_names)

    # Group words into sequences of 128
    lm_dataset_dict = lm_dataset_dict.map(group_texts, batched=True, num_proc=4)

    return lm_dataset_dict

def get_frozen_model(nlp_model, train_all_params=False):
    hf_name = get_hf_name(nlp_model)

    if 'bigbird' in nlp_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_name, attention_type='original_full').to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_name).to(device)
        
    # Whether to train or freeze all params
    for name, param in model.named_parameters():
        param.requires_grad = train_all_params

    # Unfreeze only LM head
    model.lm_head.weight.requires_grad = True

    # Print only the weights that require training
    weights_require_grad_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights_require_grad_list.append(name)
    print(f'requires_grad=True: {weights_require_grad_list}')

    return model

def get_tokenizer(nlp_model):
    hf_name = get_hf_name(nlp_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return tokenizer

def get_weights_not_equal(frozen_model, ret_model):
    weights_not_equal = []
    
    frozen_weights = {}
    for name, param in frozen_model.named_parameters():
        frozen_weights[name] = param
    
    ret_weights = {}
    for name, param in ret_model.named_parameters():
        ret_weights[name] = param
    
    for name in frozen_weights.keys():
        frozen_weight = frozen_weights[name]
        ret_weight = ret_weights[name]
        if not torch.equal(frozen_weight, ret_weight):
            weights_not_equal.append(name)

    return weights_not_equal

def calc_multi_fold_perplexity(nlp_model, output_dir):
    # Return trained model and list of perplexity results, one for each CV fold
    ret_model = None
    ret_perplexity_list = []
    
    # Get HP text data and indices for CV folds
    text_array = get_HP_text_array()
    n_words = len(text_array) # 5176
    n_folds = 4
    ind = CV_ind(n_words, n_folds=n_folds)

    for ind_num in range(n_folds):
        # Prepare model (freeze and only train LM head) and tokenizer
        model = get_frozen_model(nlp_model, train_all_params=True)
        tokenizer = get_tokenizer(nlp_model)
        
        # Prepare dataset dictionary (dict stores train, val, test Datasets)
        train_ind = ind!=ind_num
        test_ind = ind==ind_num
        train_words = text_array[train_ind]
        val_words = train_words[ int(0.8 * len(train_words)) :]
        train_words = train_words[ : int(0.8 * len(train_words)) ]
        test_words = text_array[test_ind]
        lm_dataset_dict = create_LM_dataset_dict(train_words, val_words, test_words, tokenizer)

        # Data collator for Language Modeling
        if 'long-t5' in nlp_model:    # Causal LM
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) 
        else:                       # Masked LM
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15) 

        # Train on Train dataset, evaluate on Val dataset, and use early stopping
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_strategy="epoch",       # log Training loss every epoch
            evaluation_strategy="epoch",    # log Evaluation loss every epoch
            learning_rate=2e-5,
            num_train_epochs=100,
            weight_decay=0.01,
            report_to="tensorboard",
            save_strategy='epoch',
            save_total_limit=1,             # Only save the top 1 best model
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=lm_dataset_dict["train"],
            eval_dataset=lm_dataset_dict["val"],
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        ret_model = model
        
        # Evaluate on Test dataset, and store perplexity result
        eval_results = trainer.evaluate(lm_dataset_dict["test"])
        perplexity = round(math.exp(eval_results['eval_loss']), 2)
        print(f'Perplexity for fold {ind_num}:', perplexity)
        ret_perplexity_list.append(perplexity)
        
    frozen_model = get_frozen_model(nlp_model)
    trained_weights = get_weights_not_equal(frozen_model, ret_model)
    
    return ret_perplexity_list, trained_weights
