# TurboQuant

An unofficial, end-to-end PyTorch implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) for KV cache compression during Hugging Face `transformers` generation.
Targets [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (3.09B params, 36 layers, GQA with 16 Q heads and 2 KV heads, 128-dim head, 32K context).

> [!NOTE]
> This is purely an experimental yet practical research implementation of TurboQuant and is not intended for production use. Compression happens in Python on every `update()` call, so latency scales linearly with sequence length. 

> [!WARNING]
> This implementation targets dense (non-MoE) transformer architectures. It has not been tested on Mixture-of-Experts models such as [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), where expert-specific KV routing introduces additional complexity. MoE support is planned for a future iteration.


## Key Impact

The KV cache is the primary bottleneck for scaling LLM inference. TurboQuant provides up to **5x compression** with near-zero quality loss, enabling:

*   **5x Concurrent Users**: Direct 5x reduction in serving cost per query by packing more requests per GPU.
*   **5x Context Windows**: Expand context limits within the same memory budget.
*   **Zero Calibration**: Compress on-the-fly as tokens stream in; no pre-computation required.
*   **7x Attention Speedup**: 4-bit quantization on H100 GPUs reduces HBM data movement, yielding up to 7x faster attention compute.

> [!TIP]
> At H100 compute prices (~$2-3/hr), serving 5x more users per GPU can translate to millions in infrastructure savings at scale.


## How it works

TurboQuant compresses KV cache entries online (as they arrive during generation) using a two-stage quantizer:

1. **Qmse** — Random rotation (Haar-distributed orthogonal matrix) followed by per-coordinate Lloyd-Max scalar quantization. Minimizes MSE distortion.
2. **Qprod** — 1-bit QJL (Quantized Johnson-Lindenstrauss) residual sketch on top of Qmse. Stores sign bits of the residual projected through a random Gaussian matrix, enabling unbiased inner-product estimation at attention time.

Values are quantized with Qmse only. Keys use the full Qmse + Qprod pipeline.


## Architecture

| Component | Purpose |
|---|---|
| `LloydMaxCodebook` | Solves the Lloyd-Max scalar quantizer for the marginal distribution of a random rotation's coordinates |
| `TurboQuantMSECompressor` | Rotation, quantize, pack (keys or values) |
| `TurboQuantProdCompressor` | MSE compress + QJL residual sketch (keys only) |
| `TurboQuantPaperCacheLayer` | Cache layer using Qmse+Qprod for keys, Qmse for values. Custom attention scoring with residual correction |
| `TurboQuantMSECacheLayer` | Cache layer using Qmse-only for both keys and values. Standard dot-product attention on reconstructed keys |
| `TurboQuantPaperCache` | Full cache wrapping `TurboQuantPaperCacheLayer` across all model layers |
| `TurboQuantGenerationCache` | Full cache wrapping `TurboQuantMSECacheLayer` across all model layers |
| `_patched_qwen2_forward` | Monkey-patched `Qwen2Attention.forward` that routes through TurboQuant's custom attention scoring when a quantized cache is active |

### Bit packing

Quantized indices and QJL sign bits are packed into `uint8` tensors at arbitrary bit-widths (1-8 bits). This is the main memory savings mechanism --- a 3-bit codebook index for a 128-dim head uses 48 bytes per token per head instead of 256 bytes at fp16.


## Cache modes

| Cache class | Key quantizer | Value quantizer | Default bits | Use case |
|---|---|---|---|---|
| `TurboQuantPaperCache` | Qmse + Qprod | Qmse | 3 | Paper-faithful, highest fidelity attention scores via residual correction |
| `TurboQuantGenerationCache` | Qmse only | Qmse only | 4 | Faster, no residual sketch overhead, simpler attention path |


## Target model

[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) --- a decoder-only causal LM from the Qwen2.5 series.

| Property | Value |
|---|---|
| Architecture | Transformer with RoPE, SwiGLU, RMSNorm, QKV bias, tied embeddings |
| Parameters | 3.09B (2.77B non-embedding) |
| Layers | 36 |
| Attention heads | 16 Q, 2 KV (GQA) |
| Head dimension | 128 |
| Context length | 32,768 tokens |
| Max generation | 8,192 tokens |

> [!IMPORTANT]
> Requires `transformers>=4.37.0`. Earlier versions raise `KeyError: 'qwen2'` because the Qwen2 model class was not yet registered.


## Requirements

```
torch
transformers>=5.2.0
scipy
```


## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant import (
    TurboQuantPaperCache,
    TurboQuantGenerationCache,
    patch_attention,
)

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

patch_attention(model)

cache = TurboQuantPaperCache.from_model_config(model.config, bits=3)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        past_key_values=cache,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Running the built-in evaluation

The script includes a self-contained evaluation harness that compares FP16 baseline against TurboQuant 3-bit, 4-bit, and MSE-only 4-bit configurations on a few-shot extraction task:

```bash
python turboquant.py
```


## Evaluation harness

The `__main__` block runs four configurations against a deterministic few-shot extraction prompt and checks exact-match agreement with the FP16 baseline:

| Configuration | Quantizer | Bits | Description |
|---|---|---|---|
| FP16 Baseline | None | 16 | Standard HF generation, no cache compression |
| TQ 3-bit | Qmse + Qprod | 3 | Paper cache with residual sketch |
| TQ 4-bit | Qmse + Qprod | 4 | Paper cache at higher bit-width |
| MSE 4-bit | Qmse only | 4 | Generation cache, no residual correction |

After generation, the harness prints per-configuration match status and compression statistics for layer 0.


## Credits

- TurboQuant paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) by Alibaba Cloud
