# LMPool

A personal collection of language model implementations, NLP experiments, and inference workflows.

---

## List of Projects

1. miniLLM
2. miniJamba (TODO)
3. miniSampling (TODO)

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
