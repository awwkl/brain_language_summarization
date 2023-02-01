import os
import torch
from datasets import load_dataset, load_metric
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.perplexity import get_hf_name

# Global variables are used to avoid having to set them up multiple times
rouge = load_metric("rouge")

encoder_max_length = 8192
decoder_max_length = 1024
batch_size = 2
gradient_accumulation_steps = 4

def set_global_configs(nlp_model):
    global encoder_max_length, batch_size, gradient_accumulation_steps
    if 'bigbird' in nlp_model:
        encoder_max_length = 4096   # BigBird max sequence length is 4096
        batch_size = 1              # BigBird is huge, has 32 layers
    if 'bart' in nlp_model:
        encoder_max_length = 1024   # BART max sequence length is 1024
        batch_size = 4
        gradient_accumulation_steps = 2
    if 'long-t5' in nlp_model:
        batch_size = 1              # LongT5 is large

def get_tokenizer(nlp_model):
    hf_name = get_hf_name(nlp_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    return tokenizer

def get_model(nlp_model):
    hf_name = get_hf_name(nlp_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
    return model

def set_model_configs(model):
    # set generate hyperparameters
    model.config.num_beams = 4
    model.config.max_length = 1024
    model.config.min_length = 100
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

def process_data_to_model_inputs(batch, tokenizer):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["chapter"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["summary"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch

def get_processed_train_val_datasets(tokenizer):
    booksum_dataset = load_dataset("kmfoda/booksum")
    booksum_train = booksum_dataset['train']
    booksum_val = booksum_dataset['validation']
    
    cols_to_remove=['summary_text', 'summary_url', 'summary_path', 'bid', 'summary_analysis', 'summary', 'source', 
                    'summary_name', 'summary_length', 'chapter_path', 'content', 'summary_id', 'chapter_length', 
                    'book_id', 'chapter', 'is_aggregate', 'analysis_length']
    booksum_train_processed = booksum_train.map(process_data_to_model_inputs, batched=True, fn_kwargs={"tokenizer": tokenizer},
                                                    remove_columns=cols_to_remove)
    booksum_val_processed = booksum_val.map(process_data_to_model_inputs, batched=True, fn_kwargs={"tokenizer": tokenizer},
                                                    remove_columns=cols_to_remove)

    booksum_train_processed.set_format(type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])
    booksum_val_processed.set_format(type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

    return booksum_train_processed, booksum_val_processed

def get_processed_test_dataset():
    booksum_dataset = load_dataset("kmfoda/booksum")
    booksum_test = booksum_dataset['test']
    
    cols_to_remove=['summary_text', 'summary_url', 'summary_path', 'bid', 'summary_analysis', 'summary', 'source', 
                    'summary_name', 'summary_length', 'chapter_path', 'content', 'summary_id', 'chapter_length', 
                    'book_id', 'chapter', 'is_aggregate', 'analysis_length']
    booksum_test_processed = booksum_test.map(process_data_to_model_inputs, batched=True,
                                        remove_columns=cols_to_remove)

    booksum_test_processed.set_format(type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

    return booksum_test_processed

def finetune_model_on_booksum(nlp_model, output_dir):
    set_global_configs(nlp_model)
    tokenizer = get_tokenizer(nlp_model)
    model = get_model(nlp_model)
    set_model_configs(model)
    booksum_train_processed, booksum_val_processed = get_processed_train_val_datasets(tokenizer)
    
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        # fp16_backend="apex",
        half_precision_backend="cuda_amp",
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir,
        report_to="tensorboard",
        logging_steps=100,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=booksum_train_processed,
        eval_dataset=booksum_val_processed,
    )
    trainer.train()

@torch.inference_mode()
def generate_answer(batch, nlp_model, model, tokenizer):
    print('tokenize')
    inputs_dict = tokenizer(batch["chapter"], padding="max_length", max_length=encoder_max_length, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")

    print('generate')
    if (nlp_model.startswith('led-')):
        global_attention_mask = torch.zeros_like(attention_mask)
        # put global attention on <s> token
        global_attention_mask[:, 0] = 1
        pred_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    else:
        pred_ids = model.generate(input_ids, attention_mask=attention_mask)

    print('decode')
    batch["predictions"] = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    return batch

@torch.inference_mode()
def compute_rouge(nlp_model, checkpoint_dir_or_HF_name):
    set_global_configs(nlp_model)   # This is necessary to ensure the correct encoder lengths are used
    tokenizer = get_tokenizer(nlp_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir_or_HF_name).to("cuda").half()
    set_model_configs(model)
    
    booksum_dataset = load_dataset("kmfoda/booksum")
    booksum_test = booksum_dataset['test']
    # booksum_test = booksum_test.select(range(16))    # Uncomment for testing
    
    result = booksum_test.map(generate_answer, batched=True, batch_size=4, 
                              fn_kwargs={"nlp_model": nlp_model, "model": model, "tokenizer": tokenizer})
    
    rouge_results = rouge.compute(predictions=result['predictions'], references=result['summary'])
    return rouge_results
