# Training a GPT

A comprehensive, from-scratch guide to building and training a modern decoder-only Transformer. Every component is explained with analogies, math, and heavily-annotated code.

> [!NOTE]
> This guide implements the **LLaMA 3 / Mistral / Qwen 2.5 architecture** — the best publicly-documented Transformer design. GPT-4 and Claude architectures are proprietary/undisclosed.

---

## 🗺️ Chapters

| Chapter | What You'll Learn |
|---|---|
| **[0: Overview](chapters/00_overview.md)** | What is a GPT? The big picture |
| **[1: Setup](chapters/01_setup.md)** | Install tools, GPU vs CPU, venv, PyTorch basics |
| **[2: Tokenization](chapters/02_tokenization.md)** | BPE walkthrough: how "unbelievably" becomes tokens |
| **[3: Embeddings](chapters/03_embeddings.md)** | How numbers become meaning. king − man + woman = queen |
| **[4: Positional Encoding](chapters/04_positional_encoding.md)** | RoPE: why LLaMA rotates vectors, not adds numbers |
| **[5: Attention](chapters/05_attention.md)** | ⭐ THE CORE. Q,K,V, scaling, causal mask, 8-step walkthrough |
| **[6: Transformer Block](chapters/06_transformer_block.md)** | RMSNorm, SwiGLU, residuals, pre-norm vs post-norm |
| **[7: Complete GPT Model](chapters/07_gpt_model.md)** | 124M parameter model, weight tying, logits explained |
| **[8: Training Pipeline](chapters/08_training.md)** | Cross-entropy, backprop, AdamW, cosine warmup, mixed precision |
| **[9: Inference](chapters/09_inference.md)** | KV cache, temperature, top-k/p, beam search, repetition penalty |
| **[10: Full Script](chapters/10_full_script.md)** | Runnable `main.py`: everything in one file |
| **[11: Glossary](chapters/11_glossary.md)** | Architecture provenance table, parameter breakdown |

> [!TIP]
> Start with [Chapter 0](chapters/00_overview.md) and read sequentially. Each chapter builds on the previous.

---

## 🏗️ What You'll Build

| 🧩 Component | 📝 Lines | 💡 What You'll Understand |
|---|---|---|
| **BPE Tokenizer** | ~60 | How GPT-4 splits "unbelievably" → "un" + "believ" + "ably" |
| **Embeddings** | ~30 | How "cat" and "dog" end up near each other in 768D space |
| **RoPE** | ~70 | Why LLaMA rotates vectors instead of adding position numbers |
| **Multi-Head Attention** | ~120 | The exact 8-step computation behind every modern LLM |
| **Transformer Block** | ~50 | Why residual connections are the "gradient highway" |
| **Full GPT Model** | ~200 | 124M parameter model with weight tying and pre-norm |
| **Training Pipeline** | ~250 | AdamW, cosine warmup, mixed precision, gradient accumulation |
| **Inference Engine** | ~80 | KV cache, temperature, top-k/p, beam search |


---

## 🏛️ Architecture

This guide implements the **latest publicly-documented** decoder-only Transformer:

| 🧬 Technique | 📦 Source Model | ⚡ Why It Matters |
|---|---|---|
| **RoPE** | LLaMA, Mistral, Qwen | Relative position without learned parameters |
| **RMSNorm** | LLaMA, Mistral, Gemma | 15% faster than LayerNorm, equally effective |
| **SwiGLU** | PaLM, LLaMA, Gemini | Learns which information to pass or block |
| **Pre-Norm** | GPT-3, all modern | Stable training at 100+ layers |
| **AdamW** | GPT-3+ | Better generalization than vanilla Adam |
| **BPE** | GPT-2/3/4 | Handles any text. Even unseen words and emoji |
| **Weight Tying** | GPT-2/3 | Saves 30% parameters, improves training signal |
| **Mixed Precision** | All production LLMs | 2× speed, half memory, same quality |

> [!NOTE]
> GPT-4 and Claude architectures are proprietary/undisclosed. This teaches the best publicly-confirmed architecture: what LLaMA 3, Mistral and Qwen 2.5 use.

