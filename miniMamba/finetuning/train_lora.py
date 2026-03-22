import os
from functools import partial

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

MODEL_NAME = "state-spaces/mamba-130m-hf"
TRAIN_DATASET = "Abirate/english_quotes"
OUTPUT_DIR = "./results/mamba-lora-finetuned"
LOG_DIR = "./logs"
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token


def tokenize(example, tokenizer, max_length):
    outputs = tokenizer(example["quote"], truncation=True, max_length=max_length, padding="max_length")
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def load_data(train_dataset, tokenizer, max_length):
    dataset = load_dataset("json", data_files=train_dataset)
    return dataset.map(
        partial(tokenize, max_length=max_length, tokenizer=tokenizer),
        batched=False,
        num_proc=os.cpu_count(),
    )


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

# Target_modules are Mamba-specific SSM projection layers
# Attention models use q_proj/v_proj instead.
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    max_grad_norm=0.3,
    weight_decay=0.001,
    warmup_ratio=0.03,
    logging_dir=LOG_DIR,
    logging_steps=10,
    learning_rate=2e-3,
    lr_scheduler_type="linear",
    save_total_limit=2,
    save_steps=100,
    fp16=False,
    bf16=False,
)

dataset = load_dataset(TRAIN_DATASET, split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)

if __name__ == "__main__":
    trainer.train()
    adapter_path = os.path.join(OUTPUT_DIR, "lora-adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Training complete. Adapter saved to {adapter_path}")
