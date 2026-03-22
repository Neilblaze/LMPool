# LMPool

A personal collection of language model implementations, NLP experiments, and inference workflows.

---

## List of Projects

1. miniLLM
2. miniMamba

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


# License
MIT