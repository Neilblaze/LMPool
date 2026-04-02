# LMPool

A personal collection of language model implementations, NLP experiments, and inference workflows.

---

## Index

1. miniLLM
2. miniMamba
3. AevRL
4. microAR


## Others

- Fine-Tuning LLMs (Guide)
- Self-Optimizer Inference

---

### (1) miniLLM

A minimal, readable decoder-only language model baseline incorporating practices from standard LLMs.

Designed as a quick baseline for testing architecture ideas and research experiments. The focus is on legibility and ease of modification rather than production throughput or distributed training.

See [miniLLM/README.md](miniLLM/README.md) for full details.

**Architecture highlights:**
- RMSNorm
- Rotary Position Embeddings (RoPE) with interpolation/extrapolation
- Flash Attention via `F.scaled_dot_product_attention`
- SwiGLU feed-forward
- Pre-layer normalization
- Grouped Query Attention (GQA)
- KV caching
- Multi-head Latent Attention (MLA) with compressed KV caching (DeepSeek V3 style)

**Training setup:**
- C4 dataset streamed from Hugging Face
- GPT-NeoX-20B tokenizer
- AdamW optimizer (betas 0.9, 0.95; weight decay 0.1)
- Cosine LR schedule with linear warmup
- Gradient clipping at 1.0
- Automatic Mixed Precision (AMP)
- `torch.compile()`

**Entry points:**
- `miniLLM/main.py` - standalone training script
- `miniLLM/miniLLM.ipynb` - Colab-compatible notebook

---

### (2) miniMamba

A from-scratch implementation of the Mamba selective state space model, including the full parallel scan algorithm with a custom autograd function and support for autoregressive inference.

Built for legibility and experimentation. The model can be pretrained from scratch or used as a continued pretraining starting point, with a separate LoRA fine-tuning path.

**Architecture highlights:**
- Selective SSM following the [Mamba](https://arxiv.org/abs/2312.00752) architecture (Gu & Dao, 2023)
- Custom parallel scan (`PScan`) via `torch.autograd.Function` with up/down sweep, O(log T) steps
- Depthwise causal 1D convolution as a short input filter
- Dual-branch SiLU-gated projection (x and z branches)
- S4D real initialization for the state matrix A; dt initialized via softplus inverse
- RMSNorm with optional muP scaling
- Autoregressive step-wise inference with O(1) per-step cost via RNN-style hidden state and input buffer cache
- Optional inner layer normalization (Jamba-style)
- Optional fused CUDA selective scan via `mamba_ssm`

**Pretraining setup:**
- Wikitext-2-raw-v1 by default (dataset and config fully overridable via CLI)
- Full parameter training, no PEFT
- Causal language modeling objective; text concatenated and chunked into fixed-length blocks
- HuggingFace `Trainer` with `DataCollatorForLanguageModeling`
- Cosine LR schedule with warmup, AdamW optimizer
- Automatic bf16 detection, gradient checkpointing support
- Continued pretraining from existing weights or random init from architecture config

**Fine-tuning setup:**
- LoRA via PEFT targeting Mamba SSM projection layers (`x_proj`, `in_proj`, `out_proj`, `embeddings`)
- `SFTTrainer` from trl

**Entry points:**
- `miniMamba/mamba_mini.py` - minimal model definition (Mamba, MambaBlock, PScan, RMSNorm)
- `miniMamba/mamba.py` - model definition (Mamba, MambaBlock, PScan, RMSNorm)
- `miniMamba/pscan.py` - parallel scan with custom forward and backward pass
- `miniMamba/pretraining/train.py` - pretraining script with full CLI
- `miniMamba/pretraining/pretrain.py` - minimal pretraining script
- `miniMamba/finetuning/train_lora.py` - LoRA fine-tuning script

---

### (3) AevRL

A lightweight RL stack for training language models with [GRPO](https://abderrahmanskiredj.github.io/the-illustrated-grpo) (Group Relative Policy Optimization). The main training loop is under 500 lines of code. Built to be hackable, modular, and straightforward to extend with new algorithms and environments.

The trainer runs async rollouts against a chat model served by vLLM, collects rewards from a pluggable environment, and trains a local LoRA adapter using clipped policy gradients with a KL penalty against the frozen base model. The vLLM server and PyTorch trainer time-share a single GPU via sleep/wake cycling.

See [AevRL/README.md](AevRL/README.md) for setup instructions, configuration reference, and the full training loop walkthrough.

**Algorithm**
- GRPO with group-normalized advantages (no value function needed)
- Clipped importance-weighted policy loss (PPO-clip style) with per-token assistant masking
- KL penalty against the frozen reference model using the unbiased `exp(r) - r - 1` estimator
- Pluggable algorithm interface via `Algorithm` ABC (`compute_advantages`, `loss`)

**Included environments**
- SimpleMath (basic arithmetic with `<think>/<answer>` format rewards)
- GSM8K (streamed from HuggingFace with thread-safe shuffled iteration and answer normalization)

**Training setup**
- LoRA adapter via PEFT (rank 16, alpha 32) targeting all attention and MLP projections
- Gradient checkpointing and CPU offloading to share GPU memory with vLLM
- W&B logging enabled by default
- Pydantic-validated YAML config for all hyperparameters

**Entry points:**
- [`AevRL/src/rl/train.py`](AevRL/src/rl/train.py) - training entrypoint and main loop
- [`AevRL/src/rl/rollout.py`](AevRL/src/rl/rollout.py) - async rollout collection
- [`AevRL/src/algo/grpo.py`](AevRL/src/algo/grpo.py) - GRPO implementation
- [`AevRL/configs/gsm8k.yaml`](AevRL/configs/gsm8k.yaml) - GSM8K config
- [`AevRL/configs/simple_math.yaml`](AevRL/configs/simple_math.yaml) - SimpleMath config

---

### (4) microAR

Minimal, dependency-free implementations of [Attention Residuals](https://arxiv.org/abs/2603.15031) (MoonshotAI) applied to [karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). Both variants from the paper are implemented in the same pure-Python, scalar-autograd style as the original.

In a standard transformer, the residual stream is a running sum. Attention Residuals replace this with a learned selective mix: before each sublayer, a zero-initialized projection vector scores all previous outputs via softmax, and the sublayer receives a weighted combination instead of the undifferentiated cumulative sum.

Follow [microAR/README.md](microAR/README.md) for the full walkthrough, reference pseudocode from the paper, and execution traces.

**Variants**
- Full AttnRes (FAR) tracks every individual sublayer output as a candidate. O(2*L) candidates.
- Block AttnRes (BAR) groups layers into blocks with a partial accumulator, committing block summaries at boundaries. O(blocks) candidates.

**Entry points:**
- [`microAR/micGPT_FAR.py`](microAR/micGPT_FAR.py) - Full Attention Residuals
- [`microAR/micGPT_BAR.py`](microAR/micGPT_BAR.py) - Block Attention Residuals
- [`microAR/microgpt.py`](microAR/microgpt.py) - Baseline (standard additive residual, no AttnRes)

<br/>

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)



## Others

### 🔵 Fine-Tuning LLMs (Guide)

A practical guide to supervised fine-tuning of pre-trained language models using the Hugging Face `transformers` library.

Covers the four core fine-tuning methods (SFT, CPT, DPO, RLHF), walks through a complete end-to-end training pipeline on the GSM8K math dataset using Qwen 3 (0.6B), and documents best practices around data preparation, training strategy, and common failure modes.

See [Others/Fine-Tuning/README.md](Others/Fine-Tuning/README.md) for the full guide.

**Topics covered:**
- Supervised Fine-Tuning (SFT)
- Continued Pre-Training (CPT)
- Direct Preference Optimization (DPO)
- Reinforcement Learning from Human Feedback (RLHF)
- Dataset loading and tokenization with loss masking
- Training configuration and hyperparameter guidance
- Evaluation using loss and perplexity
- Parameter-efficient fine-tuning with LoRA and NEFTune

**Entry point:**
- [`Others/Fine-Tuning/README.md`](Others/Fine-Tuning/README.md) - step-by-step tutorial with full training script

---

### 🔵 Self-Optimizer Inference

An autonomous agent loop for optimising LLM inference throughput on Apple Silicon using [MLX](https://github.com/ml-explore/mlx). Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The setup is simple. `inference.py` is the only file the agent can modify. `prepare.py` is a locked evaluation harness that benchmarks every change and enforces quality gates (perplexity and task-level sanity checks). The agent hill-climbs on generation tokens/sec, commits each experiment, and reverts anything that fails.

Tested with Claude Opus 4.6 on a MacBook Pro M4 (24GB RAM) against two models. Argmax sampling was the biggest consistent gain (+10.9% on Qwen2.5-0.5B-Instruct-4bit, +3.1% on Gemma-3-270m-it-4bit). KV cache quantisation consistently hurt, and the sanity check gate caught quality regressions that perplexity alone missed.

See [Others/SelfOptimizer-Inference/README.md](Others/SelfOptimizer-Inference/README.md) for full benchmark results and the agent protocol.

**Entry points:**
- [`Others/SelfOptimizer-Inference/inference.py`](Others/SelfOptimizer-Inference/inference.py) - MLX generation pipeline (agent-editable)
- [`Others/SelfOptimizer-Inference/prepare.py`](Others/SelfOptimizer-Inference/prepare.py) - evaluation harness (read-only)


<br/>

# License
MIT