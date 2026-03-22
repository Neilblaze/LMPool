import os
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    MambaConfig,
    MambaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "state-spaces/mamba-130m-hf"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
OUTPUT_DIR = "./results/mamba-pretrained"
LOG_DIR = "./logs"
BLOCK_SIZE = 1024

# False = continued pretraining from existing weights
# True = random init with the same architecture config.
TRAIN_FROM_SCRATCH = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

if TRAIN_FROM_SCRATCH:
    config = MambaConfig.from_pretrained(MODEL_NAME)
    model = MambaForCausalLM(config)
else:
    model = MambaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

raw_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=False)


tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=raw_dataset["train"].column_names,
)


def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    # Tail tokens that don't fill a complete block are dropped.
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [concatenated[k][i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k in concatenated.keys()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count(),
)

# mlm=False: the collator builds labels by shifting input_ids (causal LM, not masked).
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_dir=LOG_DIR,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    fp16=False,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    optim="adamw_torch",
    dataloader_num_workers=4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset.get("validation"),
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    save_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training complete. Model saved to {save_path}")
