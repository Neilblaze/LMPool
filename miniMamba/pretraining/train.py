import argparse
import os
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    MambaConfig,
    MambaForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a Mamba model on a causal language modeling objective.")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--output_dir", type=str, default="./results/mamba-pretrained")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="Initialize a fresh model using the architecture config instead of loading pretrained weights.")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to", type=str, default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.train_from_scratch:
        config = MambaConfig.from_pretrained(args.model_name)
        model = MambaForCausalLM(config)
    else:
        model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config)

    def tokenize_function(examples):
        # No truncation here; group_texts handles length by concatenating and chunking.
        return tokenizer(examples["text"], truncation=False)

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing",
    )

    block_size = args.block_size

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        # Tail tokens that don't fill a complete block are dropped.
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        return {
            k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Chunking into blocks",
    )

    # mlm=False: the collator builds labels by shifting input_ids (causal LM, not masked).
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=False,
        bf16=use_bf16,
        optim="adamw_torch",
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    save_path = os.path.join(args.output_dir, "final")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()
