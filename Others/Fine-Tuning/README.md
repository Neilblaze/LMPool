# Fine-Tuning LLMs

Training a large language model from scratch costs millions of dollars and takes months of compute time. In most practical cases, you will never need to do that. Fine-tuning lets you adapt a pre-trained model to a specific task in hours or days, using a fraction of the resources.

This guide covers the core fine-tuning techniques, walks through a complete training pipeline in Python, and highlights the practices that separate production-ready models from expensive experiments.

---

## What Is LLM Fine-Tuning?

Fine-tuning trains an existing language model on your data to improve its performance on a specific task. Pre-trained models are capable generalists, but exposing them to focused examples can turn them into reliable specialists for your use case.

Rather than building a model from scratch — which demands massive compute and data — you are giving an already-capable model a focused course in what matters to your application, whether that is medical diagnosis, customer support automation, sentiment analysis, or any other domain-specific task.

---

## How Does LLM Fine-Tuning Work?

Fine-tuning continues the training process on a pre-trained model using your own dataset. The model processes your examples, compares its outputs to the expected results, and updates its internal weights to minimize loss.

The right approach depends on your goals, available data, and compute budget. Some projects benefit from full fine-tuning, where all model parameters are updated. Others work better with parameter-efficient methods like LoRA, which modify only a small subset of parameters while preserving most of what the model already knows.

---

## Fine-Tuning Methods

### Supervised Fine-Tuning (SFT)

SFT teaches a model to match specific question-answer pairs by adjusting its weights toward those expected outputs. You need a dataset of `(Prompt, Ideal Response)` pairs. This is the right choice when you want consistent, formatted outputs — such as always responding in JSON, following a customer service script, or writing in a particular tone.

### Unsupervised Fine-Tuning (Continued Pre-Training)

This method feeds the model large amounts of raw, unlabeled text so it can learn the vocabulary and patterns of a particular domain. While technically a pre-training process known as Continued Pre-Training (CPT), it is typically applied after the initial pre-training phase. It works best when your model needs to understand specialized content it was not originally trained on — such as medical terminology, legal contracts, or a low-resource language.

### Direct Preference Optimization (DPO)

DPO trains the model to prefer better responses by showing it examples of good and bad answers to the same prompt, then adjusting weights to favor the better ones. It requires `(Prompt, Good Response, Bad Response)` triplets. Use DPO after basic instruction tuning to correct undesirable behaviors — such as hallucination, verbosity, or unsafe outputs.

### Reinforcement Learning from Human Feedback (RLHF)

RLHF involves two stages. First, you train a reward model on prompts with multiple human-ranked responses, teaching it to predict which outputs people prefer. Then, you use reinforcement learning to optimize the language model so that it generates responses the reward model scores highly. The dataset format required is `(Prompt, [Response A, Response B, ...], [Rankings])`. This approach works well for tasks where judging quality is easier than writing perfect examples, such as medical diagnosis support, legal reasoning, or other complex domains.

---

## Step-by-Step Fine-Tuning Tutorial

This tutorial walks through fine-tuning a small pre-trained model to solve word-based math problems — a task it performs poorly on by default. We use the Qwen 3 base model with 0.6 billion parameters, which already has strong natural language processing capabilities.

The approach generalizes well to other use cases: teaching specialized terminology, improving task-specific performance, or adapting to your domain.

### Prerequisites

In a new project folder, create and activate a Python virtual environment, then install the required libraries:

```bash
pip install requests datasets transformers 'transformers[torch]'
```

---

### 1. Get and Load the Dataset

Choosing the right dataset is arguably the most important step in fine-tuning. The data should directly reflect the task you want the model to perform.

Simple tasks like sentiment analysis can use basic input-output pairs. Complex tasks like instruction following or question answering require richer datasets with context, varied formats, and good coverage. Data quality and size directly affect training time and final model performance.

The easiest starting point is the [Hugging Face dataset library](https://huggingface.co/datasets), which hosts thousands of open-source datasets across domains and tasks. For specialized or high-quality data, you can purchase curated datasets or build your own by scraping publicly available sources.

For example, if you want to build a sentiment analysis model for Amazon product reviews, you could collect data from real reviews using a web scraping tool. Here is a simple example using the Oxylabs Web Scraper API:

```python
import json
import requests

# Web Scraper API parameters.
payload = {
    "source": "amazon_product",
    # Query is the ASIN of a product.
    "query": "B0DZDBWM5B",
    "parse": True,
}

# Send a request to the API and get the response.
response = requests.post(
    "https://realtime.oxylabs.io/v1/queries",
    # Visit https://dashboard.oxylabs.io to claim FREE API tokens.
    auth=("USERNAME", "PASSWORD"),
    json=payload,
)
print(response.text)

# Extract the reviews from the response.
reviews = response.json()["results"][0]["content"]["reviews"]
print(f"Found {len(reviews)} reviews")

# Save the reviews to a JSON file.
with open("reviews.json", "w") as f:
    json.dump(reviews, f, indent=2)
```

> [!NOTE]
> Feel free to use any other web scraping tool or API to collect data, but be careful about the terms of service and copyright laws.

For this tutorial, we keep things simple and skip a custom data collection pipeline. Since we are teaching the base model to solve word-based math problems, we use the `openai/gsm8k` dataset — a collection of grade-school math problems with step-by-step solutions. Load it in your Python file:

```python
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")
print(dataset["train"][0])
```

---

### 2. Tokenize the Data

Models do not process text directly — they work with numbers. Tokenization converts text into tokens (numerical representations) that the model can process. Every model has its own tokenizer trained alongside it, so use the one that matches your base model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.pad_token = tokenizer.eos_token
```

How you tokenize data shapes what the model learns. For math problems, we want the model to learn how to answer questions, not generate them. The trick is to tokenize questions and answers separately, then apply a masking technique.

Setting question tokens to `-100` tells the training process to ignore them when calculating loss. The model only learns from the answers, which makes training more focused and efficient.

```python
def tokenize_function(examples):
    input_ids_list = []
    labels_list = []

    for question, answer in zip(examples["question"], examples["answer"]):
        # Tokenize question and answer separately
        question_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        # Combine question + answer for input
        input_ids = question_tokens + answer_tokens
        # Mask question tokens with -100 so loss is only computed on the answer
        labels = [-100] * len(question_tokens) + answer_tokens

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }
```

Apply this tokenization function to both training and evaluation datasets. We filter out examples longer than 512 tokens to keep memory usage manageable and ensure the model processes complete inputs without truncation. Shuffling the training data helps the model generalize better:

```python
train_dataset = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
).filter(lambda x: len(x["input_ids"]) <= 512).shuffle(seed=42)

eval_dataset = dataset["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["test"].column_names,
).filter(lambda x: len(x["input_ids"]) <= 512)

print(f"Samples: {len(dataset['train'])} -> {len(train_dataset)} (after filtering)")
print(f"Samples: {len(dataset['test'])} -> {len(eval_dataset)} (after filtering)")
```

**Optional:** To quickly validate the full pipeline before committing to a lengthy training run, you can train on a smaller subset. Instead of the full 8,500-sample dataset, limit it to 3,000 samples to speed things up:

```python
train_dataset = train_dataset.select(range(2000))
eval_dataset = eval_dataset.select(range(1000))
```

Keep in mind that smaller datasets increase the risk of overfitting, where the model memorizes training examples instead of learning general patterns. For production use, aim for at least 5,000 training samples and tune your hyperparameters carefully.

---

### 3. Initialize the Base Model

Load the pre-trained base model that we will fine-tune:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model.config.pad_token_id = tokenizer.pad_token_id
```

---

### 4. Fine-Tune Using the Trainer

`TrainingArguments` controls how your model learns. These settings can make or break your fine-tuning run, so treat them as a starting point and adjust based on your data and hardware.

**Key parameters:**

- **Epochs** — More epochs give the model more learning opportunities, but too many lead to overfitting.
- **Batch size** — Affects memory usage and training speed. Tune based on your hardware.
- **Learning rate** — Controls how aggressively the model updates. Too high and it overshoots; too low and training stalls.
- **Weight decay** — Discourages the model from relying too heavily on any single pattern, which reduces overfitting. Set it too high and the model starts underfitting.

The configuration below is tuned for CPU training. Remove `use_cpu=True` if you have a GPU:

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./qwen-ft-math",  # Output directory for the fine-tuned model
    use_cpu=True,               # Remove or set to False to use GPU

    # Training duration
    num_train_epochs=2,         # 3 may improve reasoning at the cost of overfitting

    # Batch size and memory management
    per_device_train_batch_size=5,     # Adjust based on available memory
    per_device_eval_batch_size=5,
    gradient_accumulation_steps=4,    # Reduces memory usage

    # Learning rate and regularization
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # Evaluation and checkpointing
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Logging
    logging_steps=25,
    logging_first_step=True,
)

# Data collator handles padding and batching
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Fine-tune the base model
print("Fine-tuning started...")
trainer.train()
```

Once training completes, save the fine-tuned model and tokenizer:

```python
trainer.save_model("./qwen-ft-math/final")
tokenizer.save_pretrained("./qwen-ft-math/final")
```

---

### 5. Evaluate the Model

After fine-tuning, measure performance using two common metrics:

- **Loss** — Measures how far the model's predictions deviate from the target outputs. Lower is better.
- **Perplexity** (the exponential of loss) — Expresses the same information on a more intuitive scale. Lower values indicate higher confidence in predictions.

For production environments, consider adding metrics like BLEU or ROUGE to measure how closely generated responses match reference answers.

```python
import math

eval_results = trainer.evaluate()
print(f"Final Evaluation Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

You can also track F1 score, which measures precision and recall together — useful when both false positives and false negatives matter. The [Hugging Face course on evaluation](https://huggingface.co/learn/nlp-course/chapter3/3) is a good starting point for learning these metrics with the `transformers` library.

---

### Complete Fine-Tuning Script

Combining all the steps above, here is the full training script as a single Python file:

```python
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)


dataset = load_dataset("openai/gsm8k", "main")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.pad_token = tokenizer.eos_token


# Tokenization function adjusted for the specific dataset format
def tokenize_function(examples):
    input_ids_list = []
    labels_list = []

    for question, answer in zip(examples["question"], examples["answer"]):
        question_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
        answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        input_ids = question_tokens + answer_tokens
        labels = [-100] * len(question_tokens) + answer_tokens

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }


# Tokenize the data
train_dataset = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
).filter(lambda x: len(x["input_ids"]) <= 512).shuffle(seed=42)

eval_dataset = dataset["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["test"].column_names,
).filter(lambda x: len(x["input_ids"]) <= 512)

print(f"Samples: {len(dataset['train'])} -> {len(train_dataset)} (after filtering)")
print(f"Samples: {len(dataset['test'])} -> {len(eval_dataset)} (after filtering)")

# Optional: Use a smaller subset for faster testing
# train_dataset = train_dataset.select(range(2000))
# eval_dataset = eval_dataset.select(range(1000))


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model.config.pad_token_id = tokenizer.pad_token_id

# Configuration settings and hyperparameters for fine-tuning
training_args = TrainingArguments(
    output_dir="./qwen-ft-math",
    use_cpu=True,

    # Training duration
    num_train_epochs=2,

    # Batch size and memory management
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    gradient_accumulation_steps=4,

    # Learning rate and regularization
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # Evaluation and checkpointing
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Logging
    logging_steps=25,
    logging_first_step=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Fine-tune the base model
print("Fine-tuning started...")
trainer.train()

# Save the final model
trainer.save_model("./qwen-ft-math/final")
tokenizer.save_pretrained("./qwen-ft-math/final")

# Evaluate after fine-tuning
eval_results = trainer.evaluate()
print(f"Final Evaluation Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

Before running this, adjust the trainer configuration and hyperparameters based on what your hardware can handle.

For reference, training on a MacBook Air with an M4 chip and 16 GB of RAM took around 6.5 hours with the following settings:

- Batch size for training: 7
- Batch size for eval: 7
- Gradient accumulation steps: 5

While training, watch the evaluation loss. If it rises while training loss falls, the model is overfitting. Adjust the number of epochs, reduce the learning rate, or modify weight decay to course-correct. In a healthy run, you should see eval loss steadily decreasing — for example, from 0.496 to 0.469, with a final perplexity of 1.60.

![Training Loss Chart](https://i.postimg.cc/xj51NPCM/image.png)

---

### 6. Test the Fine-Tuned Model

Test the fine-tuned model by prompting it directly:

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    # Use `Qwen/Qwen3-0.6B` to test the base model instead
    model="./qwen-ft-math/final"
)

output = generator(
    "James has 5 apples. He buys 3 times as many. Then gives half away. How many does he have?",
    return_full_text=False
)
print(output[0]["generated_text"])
```

The side-by-side comparison below shows how the base model and the fine-tuned model respond to the same question. The correct answer is 10.

![Model Comparison](https://i.postimg.cc/qRL4nkkH/image.png)

With sampling enabled, both models can occasionally get the answer right or wrong due to randomness. Setting `do_sample=False` in the `generator()` call removes that randomness and reveals each model's true most-likely output. The base model confidently returns `-2`, while the fine-tuned model confidently returns `10`. That difference is fine-tuning working as intended.

---

## Best Practices

### Model Selection

- **Choose the right base model.** Domain-specific models and appropriate context windows reduce the amount of correction you need to do during fine-tuning.
- **Understand the model architecture.** Encoder-only models like BERT excel at classification. Decoder-only models like GPT are built for text generation. Encoder-decoder models like T5 handle transformation tasks like translation or summarization.
- **Match your model's prompt format.** If your base model was trained with specific prompt templates, use the same format during fine-tuning. Mismatched formats degrade performance noticeably.

### Data Preparation

- **Prioritize quality over quantity.** Clean, accurate examples consistently outperform large noisy datasets.
- **Keep training and evaluation data separate.** Never let the model see evaluation data during training — this is how you catch overfitting early.
- **Maintain a golden evaluation set.** Automated metrics like perplexity measure statistical confidence, not whether the model actually follows instructions correctly. Human-curated test cases are often more revealing.

### Training Strategy

- **Start with a low learning rate.** You are making targeted adjustments to an already-trained model, not building from the ground up. Aggressive learning rates risk overwriting useful pre-trained knowledge.
- **Use parameter-efficient fine-tuning.** Methods like LoRA and PEFT update roughly 1% of parameters while preserving over 90% of performance, with significantly lower memory and time requirements.
- **Target all linear layers in LoRA.** Applying LoRA to all layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.) produces models that reason more effectively, not just models that mimic style.
- **Consider NEFTune (noisy embedding fine-tuning).** Adding random noise to embeddings during training acts as a regularizer, helping prevent memorization and improving conversational quality — sometimes by 35+ percentage points.
- **Run DPO after SFT.** SFT teaches a model how to respond; DPO teaches it what a good response looks like. Using both together tends to produce significantly better results than either alone.

---

## Limitations

- **Catastrophic forgetting.** Fine-tuning can overwrite neural patterns built during pre-training, erasing general knowledge that was useful. Multi-task learning — training on your specialized data alongside general examples — helps preserve broader capabilities.

- **Overfitting on small datasets.** With limited data, a model may memorize training examples rather than learning transferable patterns, causing it to fail on inputs that differ even slightly from what it saw during training.

- **Computational cost.** Fine-tuning models with billions of parameters requires expensive GPUs, significant memory, and training runs that can span hours to weeks.

- **Bias amplification.** Pre-trained models carry biases from their training data. Fine-tuning on a poorly curated dataset can intensify those biases rather than correct them.

- **Knowledge staleness.** Information learned during fine-tuning is baked into weights. Updating that knowledge later may require retraining or integrating Retrieval-Augmented Generation (RAG). Repeated fine-tuning cycles can gradually degrade model quality.

---

## Conclusion

Fine-tuning is effective when your data is clean and your hyperparameters are well-calibrated. It works best when paired with prompt engineering — fine-tuning handles task specialization while prompt engineering shapes the model's behavior at inference time.

A practical path forward is to pick a model from Hugging Face that fits your domain, build or source a quality dataset, and run your first fine-tuning session on a small subset. Once you see promising results, scale up and experiment with LoRA, DPO, or NEFTune to push performance further. The gap between reading this guide and having a working specialized model is smaller than most people expect.