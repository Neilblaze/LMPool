# miniDeepseek

> [!NOTE]
> This project is an educational implementation of DeepSeek V3, covering the complete **Pretrain → SFT → RL** training pipeline.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Design](#2-architecture-design)
   - [Multi-head Latent Attention (MLA)](#21-multi-head-latent-attention-mla)
   - [DeepSeekMoE (Mixture of Experts)](#22-deepseekmoemixtrue-of-experts)
   - [Multi-Token Prediction (MTP)](#23-multi-token-prediction-mtp)
3. [Project Structure](#3-project-structure)
4. [Dataset Description](#4-dataset-description)
5. [Tokenizer Processing](#5-tokenizer-processing)
6. [Training Pipeline](#6-training-pipeline)
   - [Pretrain Pretraining](#61-pretrain-pretraining)
   - [SFT Supervised Fine-Tuning](#62-sft-supervised-fine-tuning)
   - [RL Reinforcement Learning](#63-rl-reinforcement-learning)
7. [RL Reinforcement Learning Details](#7-rl-reinforcement-learning-details)
   - [GRPO Algorithm Principles](#71-grpo-group-relative-policy-optimization)
   - [PPO Algorithm Principles](#72-ppo-proximal-policy-optimization)
   - [DPO Algorithm Principles](#73-dpo-direct-preference-optimization)
   - [Loss Functions and Considerations](#74-loss-functions-and-considerations)
8. [Quick Start](#8-quick-start)
9. [Configuration Guide](#9-configuration-guide)
10. [Monitoring and Visualization](#10-monitoring-and-visualization)

---

## 1. Project Overview

This project implements the core architecture and complete training pipeline of DeepSeek V3, including:

| Stage | Description | Dataset | Objective |
|------|------|--------|------|
| **Pretrain** | Language model pretraining | WikiText-2 / OpenWebText | Learn language knowledge |
| **SFT** | Supervised fine-tuning | Alpaca | Learn instruction following |
| **RL** | Reinforcement learning alignment | HH-RLHF | Align with human preferences |

### Core Innovations

1. **MLA (Multi-head Latent Attention)**: Low-rank KV compression to reduce inference memory usage
2. **DeepSeekMoE**: Mixture of experts architecture with shared experts + routed experts
3. **MTP (Multi-Token Prediction)**: Multi-token prediction as auxiliary training objective

---

## 2. Architecture Design

### 2.1 Multi-head Latent Attention (MLA)

MLA is the core attention mechanism of DeepSeek V3, reducing memory usage through low-rank KV compression.

#### Principles

Traditional Attention has KV cache size of `O(H × d_h)`, MLA compresses it to `O(d_c)`.

```
Input: x ∈ R^(B × L × D)

# KV Compression (Core Innovation)
c_kv = W_down(x)           # (B, L, d_c)     - Compress to low dimension
K = W_k_up(c_kv)           # (B, L, H, d_h)  - Expand to Key
V = W_v_up(c_kv)           # (B, L, H, d_h^v)- Expand to Value

# Query uses independent compression
c_q = W_q_down(x)          # (B, L, d_c')
Q = W_q_up(c_q)            # (B, L, H, d_h)

# Decoupled RoPE (Decoupled Position Encoding)
Q_nope, Q_rope = split(Q)  # Separate position-aware and position-free parts
K_nope, K_rope = split(K)

# Apply rotary position encoding only to rope part
Q_rope, K_rope = apply_rope(Q_rope, K_rope)

# Recombine
Q = concat(Q_nope, Q_rope)
K = concat(K_nope, K_rope)

# Standard Attention
Output = softmax(QK^T / √d_h) · V
```

#### Key Parameters

| Parameter | Meaning | Default Value |
|------|------|--------|
| `kv_lora_rank` | KV compression dimension d_c | 64 |
| `q_lora_rank` | Q compression dimension d_c' | 96 |
| `qk_nope_head_dim` | Non-RoPE head dimension | 32 |
| `qk_rope_head_dim` | RoPE head dimension | 32 |
| `v_head_dim` | Value head dimension | 64 |

#### Code Location

- Implementation: [attention.py](deepseek/model/attention.py) - `MultiHeadLatentAttention` class

---

### 2.2 DeepSeekMoE（Mixture of Experts）

DeepSeekMoE combines shared experts and routed experts, ensuring both general knowledge and specialized capabilities.

#### Architecture

```
Input: x ∈ R^(B × L × D)

# 1. Shared Experts - Always activated
shared_out = Σ expert_s(x) / n_shared

# 2. Routed Experts - Top-K selection
router_probs = softmax(gate(x))           # (B, L, N) routing probabilities
top_k_probs, top_k_idx = topk(router_probs, K)  # Select Top-K experts
routed_out = Σ (prob_i × expert_i(x))     # Weighted output

# 3. Final output
output = shared_out + routed_scaling_factor × routed_out
```

#### Load Balancing Loss

To prevent imbalanced expert usage, an auxiliary loss is introduced:

```
L_aux = α × N × Σ(f_i × P_i)

Where:
- f_i: Proportion of tokens received by expert i
- P_i: Average routing probability of expert i
- α: Loss coefficient (default 0.001)
- N: Total number of experts
```

#### Key Parameters

| Parameter | Meaning | Default Value |
|------|------|--------|
| `num_experts` | Total number of routed experts N | 16 |
| `num_experts_per_tok` | Number of experts activated per token K | 2 |
| `num_shared_experts` | Number of shared experts | 2 |
| `expert_hidden_size` | Expert FFN hidden dimension | 768 |
| `aux_loss_alpha` | Auxiliary loss coefficient | 0.001 |

#### Code Location

- Implementation: [model.py](deepseek/model/model.py) - `DeepSeekMoE`, `MoEGate`, `Expert` classes

---

### 2.3 Multi-Token Prediction (MTP)

MTP predicts multiple future tokens simultaneously as an auxiliary training objective, while also supporting speculative decoding during inference.

#### Principles

```
Input: hidden_states ∈ R^(B × L × D)

# For each prediction depth d ∈ [1, D_predict]
for d in range(1, num_predict_tokens + 1):
    # Independent projection layer
    h_d = projection_d(hidden_states)
    h_d = layer_norm_d(h_d)
    logits_d = output_head_d(h_d)  # Predict token at position i+d

# Calculate MTP Loss during training
mtp_loss = Σ CE(logits_d[:, :-d-1], labels[:, d+1:])
total_loss = lm_loss + mtp_weight × mtp_loss
```

#### Key Parameters

| Parameter | Meaning | Default Value |
|------|------|--------|
| `num_predict_tokens` | Number of additional tokens to predict | 2 |
| `mtp_loss_weight` | MTP loss weight | 0.3 |

#### Code Location

- Implementation: [model.py](deepseek/model/model.py) - `MTPHead` class

---

## 3. Project Structure

```
miniDeepseek/
├── deepseek/                    # Core package directory
│   ├── __init__.py              # Package exports
│   ├── model/                   # Model module
│   │   ├── __init__.py
│   │   ├── attention.py         # MLA attention implementation
│   │   └── model.py             # DeepSeek V3 model main body
│   ├── data/                    # Data module
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset processing (Pretrain/SFT/RL)
│   │   └── rl_dataset.py        # RL-specific dataset
│   ├── training/                # Training module
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Trainer (Pretrain/SFT/GRPO)
│   │   ├── rl_trainer_base.py         # RL training base class
│   │   └── rl_trainer_algorithms.py   # RL algorithm implementations (GRPO/PPO)
│   └── utils/                         # Utility module
│       ├── __init__.py
│       └── logger.py            # Logging module (colored output, multi-level logging)
├── configs/                     # Configuration file directory
│   ├── config_default.yaml      # Default configuration (small dataset)
│   └── config_large.yaml        # Large-scale training configuration
├── scripts/                     # Scripts directory
│   ├── run.sh                   # Convenient run script
│   └── run_pretrain.sh          # Pretraining-specific script
├── tests/                       # Test directory
│   ├── __init__.py
│   ├── test_all.py              # Test suite
│   └── test_rl.py               # RL tests
├── config.py                    # Configuration management (all config class definitions)
├── train.py                     # Training entry script
├── rl_train.py                  # RL training entry
├── inference.py                 # Inference and generation
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 4. Dataset Description

### 4.1 Pretrain Dataset

| Dataset | Size | Parameter | Usage |
|--------|------|------|------|
| **WikiText-2** | ~13MB | `--dataset_scale small` | Quick testing/experiments |
| **OpenWebText** | ~40GB | `--dataset_scale large` | Formal training |

#### WikiText-2 Format
Raw text, each line is a paragraph:
```text
= Valkyria Chronicles III =
Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場の...
```

#### OpenWebText Format
Text content from Reddit external links:
```python
{
    "text": "The full article content..."
}
```

### 4.2 SFT Dataset

| Dataset | Size | Format |
|--------|------|------|
| **Alpaca** | ~52K samples | instruction-input-output |

#### Alpaca Format
```json
{
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
}
```

#### Formatting Template
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### 4.3 RL Dataset

| Dataset | Size | Format |
|--------|------|------|
| **HH-RLHF** | ~170K | chosen/rejected pairs |

#### HH-RLHF Format
```json
{
    "chosen": "Human: What is...\n\nAssistant: The answer is...",
    "rejected": "Human: What is...\n\nAssistant: I don't know..."
}
```

---

## 5. Tokenizer Processing

This project uses GPT-2 Tokenizer (other HuggingFace tokenizers can be configured).

### 5.1 Loading Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set padding token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

### 5.2 Pretrain Data Processing

```python
# 1. Concatenate all text
all_text = " ".join(texts)

# 2. Tokenize
tokens = tokenizer.encode(all_text, add_special_tokens=False)

# 3. Split into fixed-length sequences
for i in range(0, len(tokens) - max_seq_length, max_seq_length):
    chunk = tokens[i:i + max_seq_length]
    examples.append({
        'input_ids': torch.tensor(chunk),
        'attention_mask': torch.ones(len(chunk)),
        'labels': torch.tensor(chunk),  # Autoregressive, labels = input_ids
    })
```

### 5.3 SFT Data Processing

```python
# 1. Format prompt and full text
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
full_text = prompt + output

# 2. Tokenize
prompt_ids = tokenizer.encode(prompt)
full_ids = tokenizer.encode(full_text)

# 3. Create labels (prompt part is -100, not calculated in loss)
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

### 5.4 Key Configuration

```yaml
data:
  tokenizer_name: "gpt2"           # Can be changed to other tokenizers
  pretrain_max_seq_length: 512     # Pretraining sequence length
  sft_max_seq_length: 512          # SFT sequence length
  rl_max_seq_length: 256           # RL sequence length
```

---

## 6. Training Pipeline

### 6.1 Pretrain Pretraining

#### Objective
Learn the foundational capabilities of language models: grammar, knowledge, reasoning.

#### Loss Function
```python
# Next-Token Prediction Loss
loss = CrossEntropyLoss(logits[:, :-1], labels[:, 1:])

# + MTP Loss (if enabled)
for d in range(1, num_predict_tokens + 1):
    mtp_loss += CrossEntropyLoss(mtp_logits_d[:, :-d-1], labels[:, d+1:])
loss += mtp_weight * mtp_loss / num_predict_tokens

# + MoE Auxiliary Loss (if enabled)
loss += aux_loss
```

#### Run Commands

```bash
# Quick test with small dataset (WikiText-2, ~13MB)
python train.py --mode pretrain --dataset_scale small --test

# Complete training with small dataset
python train.py --mode pretrain --dataset_scale small

# Large dataset training (OpenWebText, ~10GB)
python train.py --mode pretrain --dataset_scale large

# Using run.sh script
./run.sh pretrain          # Small dataset
./run.sh pretrain-large    # Large dataset
./run.sh pretrain-test     # Quick test
```

#### Key Parameters

```yaml
pretraining:
  batch_size: 16
  learning_rate: 3e-4
  max_steps: 5000
  warmup_steps: 200
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
```

---

### 6.2 SFT Supervised Fine-Tuning

#### Objective
Learn to follow instructions and generate helpful responses.

#### Loss Function
```python
# Calculate loss only on response part
# Prompt part in labels is set to -100
loss = CrossEntropyLoss(logits, labels, ignore_index=-100)
```

#### Run Commands

```bash
# Start SFT from pretrained checkpoint
python train.py --mode sft --checkpoint checkpoints/pretrain/best.pt

# Using run.sh
./run.sh sft checkpoints/pretrain/best.pt
./run.sh sft-test  # Quick test
```

#### Key Parameters

```yaml
sft:
  batch_size: 8
  learning_rate: 2e-5      # Smaller than pretraining
  max_steps: 2000
  warmup_ratio: 0.03
  weight_decay: 0.0        # SFT typically doesn't use weight decay
```

---

### 6.3 RL Reinforcement Learning

#### Objective
Align the model with human preferences, generating more helpful and safer responses.

#### Supported Algorithms

| Algorithm | Type | Characteristics |
|------|------|------|
| **GRPO** | Online | DeepSeek style, group relative advantage |
| **PPO** | Online | Classic RLHF, requires value function |
| **DPO** | Offline | Direct preference optimization, no reward model needed |

#### Run Commands

```bash
# GRPO (default)
python train.py --mode rl --checkpoint checkpoints/sft/best.pt

# Specify algorithm
python rl_train.py --algorithm grpo --checkpoint checkpoints/sft/best.pt
python rl_train.py --algorithm ppo --checkpoint checkpoints/sft/best.pt
python rl_train.py --algorithm dpo --checkpoint checkpoints/sft/best.pt

# Using run.sh
./run.sh rl checkpoints/sft/best.pt
./run.sh rl-test
```

---

## 7. RL Reinforcement Learning Details

### 7.1 GRPO (Group Relative Policy Optimization)

GRPO is an RL algorithm proposed by DeepSeek that doesn't require learning a reward model and uses group relative advantage.

#### Algorithm Flow

```
For each prompt x:
    1. Generate G responses {y_1, y_2, ..., y_G}
    2. Calculate reward for each response r_i = R(x, y_i)
    3. Calculate group relative advantage:
       A_i = (r_i - mean(r)) / (std(r) + ε)
    4. Calculate policy gradient loss:
       L_PG = -E[A_i × log π(y_i|x)]
    5. Calculate KL penalty:
       L_KL = β × KL(π || π_ref)
    6. Total loss:
       L = L_PG + L_KL
```

#### Core Code

```python
# Group relative advantage normalization
rewards_t = torch.tensor(rewards)
mean_r = rewards_t.mean()
std_r = rewards_t.std() + 1e-8
advantages = (rewards_t - mean_r) / std_r

# Policy Gradient Loss
for adv, log_prob in zip(advantages, log_probs):
    pg_loss += -adv * log_prob.mean()

# KL Penalty
kl = (policy_logps - ref_logps).mean()
loss = pg_loss + kl_coef * kl
```

#### Key Parameters

| Parameter | Meaning | Default Value | Description |
|------|------|--------|------|
| `group_size` | Number of responses generated per prompt | 4 | Larger values give better variance estimation but higher computation |
| `kl_coef` | KL penalty coefficient β | 0.1 | Prevents deviating too far from reference model |
| `temperature` | Sampling temperature | 0.7 | Controls generation diversity |

---

### 7.2 PPO (Proximal Policy Optimization)

PPO is a classic RLHF algorithm that uses a value function to estimate advantage.

#### Algorithm Flow

```
1. Rollout: Generate responses, calculate reward
2. Calculate GAE (Generalized Advantage Estimation):
   δ_t = r_t + γ V(s_{t+1}) - V(s_t)
   A_t = Σ (γλ)^k δ_{t+k}
3. PPO Update (multiple epochs):
   a. Calculate probability ratio: ρ = π(a|s) / π_old(a|s)
   b. Clipped surrogate objective:
      L_clip = min(ρ A, clip(ρ, 1-ε, 1+ε) A)
   c. Value function loss:
      L_VF = MSE(V(s), R_t)
   d. Entropy bonus:
      H = -Σ π log π
   e. Total loss:
      L = -L_clip + c1 × L_VF - c2 × H
```

#### Key Parameters

| Parameter | Meaning | Default Value | Description |
|------|------|--------|------|
| `clip_range` | PPO clipping range ε | 0.2 | Limits policy update magnitude |
| `value_coef` | Value loss coefficient c1 | 0.5 | |
| `entropy_coef` | Entropy bonus coefficient c2 | 0.01 | Encourages exploration |
| `gae_lambda` | GAE λ parameter | 0.95 | Variance-bias tradeoff |
| `ppo_epochs` | PPO update epochs per batch | 4 | |
| `target_kl` | KL early stopping threshold | 0.02 | Stop updating if exceeded |

---

### 7.3 DPO (Direct Preference Optimization)

DPO learns directly from preference data without an explicit reward model.

#### Algorithm Principles

```
Given preference data (x, y_w, y_l), where y_w is the human-preferred response and y_l is the non-preferred response

DPO Loss:
L_DPO = -E[log σ(β × (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

Simplified to:
L_DPO = -E[log σ(β × (r_w - r_l))]

Where:
r = log π(y|x) - log π_ref(y|x)  # Implicit reward
```

#### Core Code

```python
# Calculate policy and reference log probabilities
policy_logps_w = compute_log_probs(model, y_w)
policy_logps_l = compute_log_probs(model, y_l)
ref_logps_w = compute_log_probs(ref_model, y_w)
ref_logps_l = compute_log_probs(ref_model, y_l)

# Calculate reward margin
logits_w = policy_logps_w - ref_logps_w
logits_l = policy_logps_l - ref_logps_l
logits_diff = logits_w - logits_l

# DPO Loss
loss = -F.logsigmoid(beta * logits_diff).mean()
```

#### Key Parameters

| Parameter | Meaning | Default Value | Description |
|------|------|--------|------|
| `dpo_beta` | Temperature parameter β | 0.1 | Larger values make policy changes more aggressive |
| `dpo_label_smoothing` | Label smoothing | 0.0 | Increases robustness |

---

### 7.4 Loss Functions and Considerations

#### Main Loss Components in RL Training

| Loss | Formula | Purpose |
|------|------|------|
| **Policy Gradient** | `-A × log π(y|x)` | Increase probability of high-reward responses |
| **KL Penalty** | `β × KL(π || π_ref)` | Prevent deviating too far from reference model |
| **Value Loss** | `MSE(V, R)` | Accurately estimate state value |
| **Entropy Bonus** | `-H(π)` | Encourage exploration |

#### Training Considerations

1. **Learning Rate**: RL stage uses very small learning rate (~5e-7) to prevent model capability degradation

2. **KL Control**:
   - Monitor KL divergence; too large indicates policy changing too fast
   - Use target_kl early stopping mechanism
   - Adjust kl_coef appropriately

3. **Reward Design**:
   - This project uses rule-based reward (length, coherence, repetition penalty, etc.)
   - Production environments should use learned reward models

4. **Reward Hacking**:
   - Model may find loopholes in the reward
   - Use diverse reward signals
   - Maintain KL constraints

5. **Training Stability**:
   - Use gradient clipping `max_grad_norm: 1.0`
   - Use gradient accumulation for smooth updates
   - Monitor reward and loss curves

6. **Reference Model**:
   - Keep frozen, don't update parameters
   - Used to calculate KL divergence
   - Prevents model degradation

---

## 8. Quick Start

### 8.1 Install Dependencies

```bash
cd learn/deepseek_v3
pip install -r requirements.txt
chmod +x run.sh
```

### 8.2 Run Tests

```bash
# Test model and training pipeline
./run.sh test-quick

# Complete test
./run.sh test
```

### 8.3 Complete Training Pipeline

```bash
# 1. Pretrain (quick validation with small dataset)
./run.sh pretrain-test

# Or complete pretraining
./run.sh pretrain

# 2. SFT
./run.sh sft checkpoints/pretrain/best.pt

# 3. RL (GRPO)
./run.sh rl checkpoints/sft/best.pt

# 4. Inference
./run.sh inference checkpoints/rl/best.pt

# 5. Interactive chat
./run.sh chat checkpoints/rl/best.pt
```

### 8.4 One-Click Complete Pipeline

```bash
# Quick test of entire pipeline
./run.sh full-test

# Complete training pipeline
./run.sh full
```

---

## 9. Configuration Guide

### 9.1 Configuration Files

| File | Purpose |
|------|------|
| `config_default.yaml` | Default configuration (small dataset) |
| `config_large.yaml` | Large-scale training configuration |

### 9.2 Command Line Parameters

```bash
python train.py \
    --mode pretrain|sft|rl \     # Training mode
    --dataset_scale small|large \ # Dataset scale
    --config config.yaml \        # Configuration file
    --checkpoint path/to/ckpt \   # Load checkpoint
    --device auto|cuda|mps|cpu \  # Device
    --test                        # Quick test mode
```

### 9.3 Key Configuration Items

```yaml
model:
  hidden_size: 512           # Model dimension
  num_hidden_layers: 6       # Number of Transformer layers
  num_attention_heads: 8     # Number of attention heads
  
  moe:
    enabled: true
    num_experts: 16          # Number of experts
    num_experts_per_tok: 2   # Experts activated per token
    
  mtp:
    enabled: true
    num_predict_tokens: 2    # Additional tokens to predict

pretraining:
  batch_size: 16
  learning_rate: 3e-4
  max_steps: 5000

sft:
  batch_size: 8
  learning_rate: 2e-5

rl:
  algorithm: "grpo"          # grpo, ppo, dpo
  group_size: 4
  kl_coef: 0.1
```

---

## 10. Monitoring and Visualization

### 10.1 TensorBoard

```bash
# Start TensorBoard
./run.sh tensorboard

# Or start manually
tensorboard --logdir=runs --port=6006
```

### 10.2 Visualization Content

| Category | Content |
|------|------|
| **Loss** | train_loss, val_loss, perplexity |
| **Learning** | learning_rate, grad_norm |
| **Speed** | tokens/sec, samples/sec, steps/sec |
| **Attention** | Attention weight heatmaps |
| **MoE** | Expert usage distribution, routing entropy |
| **Generation** | Generated text samples |
| **RL** | reward, kl_divergence, policy_loss |

### 10.3 Training Log Example

```
┌──────────────────────────────────────────────────────────────────────┐
│ Step:    100/5000 [██████░░░░░░░░░░░░░░░░░░░░░░░░]  2.0%             │
├──────────────────────────────────────────────────────────────────────┤
│ Loss:   6.2344  (smoothed:   6.3011)                                 │
│ LR: 2.85e-04  Grad norm:   0.8721                                    │
│ Epoch: 1                                                             │
├──────────────────────────────────────────────────────────────────────┤
│ Speed:    12537 tok/s   98.5 samples/s   6.16 steps/s                │
│ Time:      16.5s  ETA:    13.2m                                      │
│ Tokens:      203,779                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## References

- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
