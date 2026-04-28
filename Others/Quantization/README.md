# Quantization

This directory contains minimal, readable implementations and reference pipelines for modern neural network quantization techniques. The focus is on establishing robust baselines for post-training compression, mixed-precision inference, and parameter-efficient fine-tuning.

> [!NOTE]
> Most implementations require external dependencies (`transformers`, `bitsandbytes`, `autoawq`, `auto-gptq`). Ensure your environment is configured correctly before executing the API-driven examples. And extended list of quantization techniques and their comparisons can be found in the official [HF blog](https://huggingface.co/docs/transformers/quantization/overview).

--- 

### 🔵 Techniques Overview

| Method                                      | Type               | Bits | Key Strength           |
|---------------------------------------------|--------------------|------|------------------------|
| [PTQ](./PTQ/ptq.py)                         | Dynamic PTQ        | 8/4  | Baseline inference     |
| [QAT](./QAT/qat.py)                         | Training           | 8/4  | Preserves accuracy     |
| [GPTQ](./GPTQ/gptq.py)                      | One-shot PTQ       | 4/8  | Standard weight int4   |
| [AWQ](./AWQ/awq.py)                         | Activation PTQ     | 4    | Retains outliers       |
| [SmoothQuant](./SmoothQuant/smoothquant.py) | Mathematical       | 8    | Stable W8A8 compute    |
| [LLM.int8()](./LLM_int8/llm_int8.py)        | Mixed Runtime      | 8    | Handles vector spikes  |
| [NF4](./NF4/nf4.py)                         | Data Type          | 4    | Normal distribution    |
| [FP8](./FP8/fp8.py)                         | Data Type          | 8    | Hardware optimized     |
| [SpQR](./SpQR/spqr.py)                      | Sparse-Quantized   | 4    | Near-lossless int4     |
| [QLoRA](./QLoRA/qlora.py)                   | Adapter Tuning     | 4    | Resource efficient     |

<br/>

### 🔵 Implementations

#### 🟢 PTQ (Post-Training Quantization)
Reduces the memory footprint of a trained model by dynamically or statically converting FP32 weights and activations to lower precision formats (like INT8) during inference, avoiding the overhead of retraining.

#### 🟢 QAT (Quantization-Aware Training)
Simulates the rounding and scaling effects of lower precision formats during the forward and backward training passes. This mathematically prepares the network to withstand quantization noise, yielding much higher fidelity than simple PTQ.

#### 🟢 GPTQ
An approximate second-order method for highly accurate, one-shot weight quantization. Widely considered the industry standard for squeezing massive LLMs into 4-bit envelopes with minimal perplexity degradation.

#### 🟢 AWQ (Activation-aware Weight Quantization)
Observes empirical activation distributions to identify "salient" structural weights. By protecting a small fraction of critical weights, AWQ achieves superior zero-shot robustness over blind magnitude-based weight quantization.

#### 🟢 SmoothQuant
A mathematically elegant technique that mitigates catastrophic activation outliers in large models. It migrates the difficulty of quantization away from the highly-variable activation tensors and into the relatively stable weight matrices.

> [!TIP]
> SmoothQuant is highly recommended if you are attempting to establish stable, high-throughput W8A8 (Weight 8-bit, Activation 8-bit) inference pipelines.

#### 🟢 LLM.int8()
A mixed-precision runtime architecture that intercepts severe activation outliers dynamically. The bulk of the matrix multiplication is executed efficiently in INT8, while extreme outlier channels are routed through standard FP16 pathways.

#### 🟢 NF4 (NormalFloat4)
An information-theoretically optimal precision format specifically engineered for normally-distributed neural network weights. It provides better empirical accuracy per bit compared to standard 4-bit floats or integers.

#### 🟢 FP8
Leverages native 8-bit floating point formats (like `e4m3fn` and `e5m2`) to compress tensors. With expanding hardware support across modern microarchitectures, FP8 serves as the default high-performance datatype for modern training and inference.

#### 🟢 SpQR (Sparse-Quantized Representation)
Isolates critical structural outliers into a high-precision sparse matrix, allowing the dense standard weights to be aggressively compressed (often down to 3 or 4 bits) with near-lossless fidelity at scale.

#### 🟢 QLoRA
Fuses NF4 quantization with Low-Rank Adaptation. The heavy base model is frozen in 4-bit precision while lightweight trainable adapters capture domain-specific gradients, unlocking finetuning for massive models on consumer hardware.

---

> [!NOTE]
> The scripts provided here are structured for immediate legibility and testing. Production deployments should wrap these configurations within dedicated serving frameworks like `vLLM` or `TensorRT-LLM`.

> [!TIP]
> For more such papers related to quantization techniques, visit [Awesome-LLM-Quantization](https://github.com/pprp/Awesome-LLM-Quantization). Also, for more in-depth understanding of different quantization concepts, refer to the official [HF Concept Guide](https://huggingface.co/docs/transformers/quantization/concept_guide).