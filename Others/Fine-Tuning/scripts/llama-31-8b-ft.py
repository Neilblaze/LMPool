#!/usr/bin/env python3
# This is a script for finetuning Llama 3.1 8B with QLoRA on 2 RTX 5090s

import os
import torch
import subprocess
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    __version__ as transformers_version,
)
from trl import SFTTrainer, __version__ as trl_version
from datasets import __version__ as datasets_version

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# ------------------------------------------------------------------
# Check library versions
# ------------------------------------------------------------------
print(f"TRL version: {trl_version}")
print(f"Transformers version: {transformers_version}")
print(f"Datasets version: {datasets_version}")

# ------------------------------------------------------------------
# Hugging Face login (using environment variable)
# ------------------------------------------------------------------
os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")  # Fallback to empty string
if not os.environ["HUGGINGFACE_TOKEN"]:
    raise RuntimeError("HUGGINGFACE_TOKEN not set. Run `huggingface-cli login` or set the environment variable.")
    
# ------------------------------------------------------------------
# Environment & hardware
# ------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use both GPUs
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence tokenizer warnings

# Verify CUDA and GPU setup
if not torch.cuda.is_available():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    try:
        print(f"NVIDIA Driver Info: {subprocess.check_output('nvidia-smi', shell=True).decode()}")
    except:
        print("nvidia-smi command failed. Check NVIDIA drivers.")
    raise RuntimeError("CUDA not available. Ensure NVIDIA drivers, CUDA toolkit, and PyTorch with CUDA support are installed.")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
torch.cuda.set_device(0)  # Set GPU 0 as the default for all subsequent CUDA operations


# ------------------------------------------------------------------
# Configurable hyper-parameters
# ------------------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
#m-newhauser/senator-tweets
DATASET_NAME = "timdettmers/openassistant-guanaco"
#DATASET_NAME = "fka/awesome-chatgpt-prompts"
OUTPUT_DIR = "./finetuned_llama3_1_8b"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # Conservative for 32GB VRAM
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 1 * 2 * 8 = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024  # Balanced for memory and performance - Shorter sequences may truncate longer conversations, slightly reducing model quality for multi-turn dialogs

# ------------------------------------------------------------------
# 4-bit quantization (QLoRA)
# ------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.bfloat16, - previous
    bnb_4bit_compute_dtype=torch.float8_e4m3fn # supposedly optimized
)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model {MODEL_NAME}: {e}. Ensure you have access and a valid token.")

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# ------------------------------------------------------------------
# LoRA config
# ------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# ------------------------------------------------------------------
# Tokenizer                                                     
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=os.environ["HUGGINGFACE_TOKEN"],
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
try:
    dataset = load_dataset(DATASET_NAME, split="train")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}. Check dataset name and connectivity.")

# Inspect dataset
print(f"Dataset columns: {dataset.column_names}")
#print(f"Sample example: {dataset[0]}")

# Use the existing "text" column directly
# Optional: Custom formatting if needed
# def format_example(example):
#     return {"text": example["text"]}  # Already formatted with ### Human: and ### Assistant:


# ------------------------------------------------------------------
# Training arguments
# ------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="paged_adamw_8bit",
    learning_rate=LEARNING_RATE,
    bf16=True,
    fp16=False,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="no",
    report_to="tensorboard",
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
)

# ------------------------------------------------------------------
# SFTTrainer
# ------------------------------------------------------------------
try:
    # Initialize SFTTrainer with minimal, widely-compatible kwargs.
    # Some versions of `trl` do not accept `tokenizer` or `dataset_text_field`
    # as constructor kwargs — pass the core args and attach optional pieces
    # afterwards to support more versions.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,           # LoRA config
        # packing=True,                    # optional – only if you have a compatible version
        # formatting_func=format_example,  # optional custom formatting
    )
    # Attach tokenizer and dataset/max seq info if the trainer instance
    # (or downstream code) expects them — setting attributes is safe even
    # if the class doesn't define them in __init__.
    try:
        if not hasattr(trainer, "tokenizer"):
            trainer.tokenizer = tokenizer
        if not hasattr(trainer, "dataset_text_field"):
            trainer.dataset_text_field = "text"
        if not hasattr(trainer, "max_seq_length"):
            trainer.max_seq_length = MAX_SEQ_LENGTH
    except Exception:
        # Non-fatal: silently continue if setting attributes fails.
        pass
except Exception as e:
    raise RuntimeError(f"Failed to initialize SFTTrainer: {e}. Check trl version and arguments.")


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
try:
    trainer.train()
except RuntimeError as e:
    print(f"Training failed: {e}. Try reducing batch size or max_seq_length.")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nFinetuning finished! Model saved to {OUTPUT_DIR}")

# THIS IS ADDED TO TRY TO CONVERT RIGHT AFTER THIS IS DONE - NOT TESTED YET!
# --------------------------------------------------------------------------

# After saving, convert model to GGUF format using llama.cpp tools:
# Step 1: Install llama.cpp (if not already installed)
# Step 2: Run convert.py script from transformers library
# Step 3: Use gguf tool for conversion

import subprocess

# Convert to GGUF
try:
    subprocess.run([
        "python",
        "-m",
        "transformers.convert_model_to_gguf",
        "--model_name", MODEL_NAME,
        "--output_dir", OUTPUT_DIR,
        "--quantization_type", "4bit"
    ], check=True, shell=True)
except Exception as e:
    print(f"Convert failed: {e}. Check llama.cpp and transformers library.")