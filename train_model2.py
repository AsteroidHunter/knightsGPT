"""
@author: akash
"""

from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, load_dataset
import polars as pl
import numpy as np
import pyarrow
import argparse
import torch

### ADDING ARGUMENTS TO PASS
parser = argparse.ArgumentParser()

parser.add_argument(
    '--percent_sample',
    type=float,
    help="Percentage of the training and validation sets to randomly sample",
    required=True
)

parser.add_argument(
    '--dataset_size',
    type=int,
    help="Length of the training dataset",
    required=True
)

args = parser.parse_args()


### CONFIGURING GPT-2 VARIANT
config = GPT2Config(
    vocab_size=64,   
    n_positions=64,  
    n_ctx=64,        
    n_embd=768,      
    n_layer=8,       
    n_head=16,       
    resid_pdrop=0.1, 
    embd_pdrop=0.1,
    attn_pdrop=0.1
)

model = GPT2LMHeadModel(config)

### LOADING TRAINING AND VALIDATION SET
train_small_dataset = load_dataset(
    "parquet", 
    data_files=f"./data/sampled/training_final_8x8_{int(args.percent_sample * 100)}_percent.parquet",
    split="train",
    streaming=True
)

val_small_dataset = pl.read_parquet(f"./data/sampled/val_final_8x8_{int(args.percent_sample * 100)}_percent.parquet")
val_small_dataset = Dataset(val_small_dataset.to_arrow())


print(f"Streaming set up for the {int(args.percent_sample * 100)}% sampled datasets done")

### TRAINING BLOCK

num_gpus = 2
batch_size = 128
training_epochs = 1
num_gradient_accumulation = 1

# computing the estimated number of total steps
max_steps = (args.dataset_size // (num_gpus * batch_size)) * training_epochs
max_steps //= num_gradient_accumulation


training_args = TrainingArguments(
    output_dir=f"./trained{int(args.percent_sample * 100)}/results/",
    overwrite_output_dir=True,
    eval_strategy="steps",
    num_train_epochs=training_epochs,
    max_steps=max_steps,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=num_gradient_accumulation,
    # metric_for_best_model="eval_loss",
    # greater_is_better=False,
    # learning_rate=5e-4,
    learning_rate=1e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    weight_decay=0.1,
    eval_steps=1000,
    save_steps=1000,
    logging_dir=f"./trained{int(args.percent_sample * 100)}/logs",
    logging_steps=100,
    fp16=True,
    ddp_find_unused_parameters=False,
    
    # dataloader_num_workers=1,
    # dataloader_pin_memory=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_small_dataset,
    eval_dataset=val_small_dataset,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train()
