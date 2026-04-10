# Speculative Decoding

An implementation of [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2302.01318) (Leviathan et al., 2023) with rejection sampling that provably preserves the target model's output distribution.

A small draft model (Qwen3-0.6B) proposes `gamma` tokens per step, the target model (Qwen3-4B) verifies them in a single forward pass, and a rejection sampling scheme accepts or rejects each token — guaranteeing the final output is distributionally identical to sampling from the target alone.

> [!NOTE]
> This is a research implementation. The draft and target models must share the same vocabulary for the acceptance criterion to be valid.


## How it works

Standard autoregressive decoding is bottlenecked by sequential forward passes through the target model. Speculative decoding amortizes this cost:

1. **Draft**: The draft model generates `gamma` candidate tokens autoregressively (cheap, small model).
2. **Verify**: The target model scores all `gamma` candidates + the preceding token in a single forward pass.
3. **Accept/Reject**: Each draft token is accepted with probability `min(p_target / p_draft, 1)`. On the first rejection, a corrective token is sampled from `clamp(p_target - p_draft, 0)` (re-normalized). If all `gamma` tokens are accepted, a bonus token is sampled from the target's next-position distribution.

This yields 1 to `gamma + 1` tokens per target forward pass, with the expected gain proportional to how well the draft model approximates the target.


## Architecture

| Component | Purpose |
|---|---|
| `SpeculativeConfig` | Dataclass holding decoding hyperparameters (gamma, temperature, top-k/p, max tokens) |
| `SpeculativeDecoder` | Core loop: prefill both models, then iterate draft-verify-accept cycles |
| `_logits_to_probs` | Temperature scaling + top-k + nucleus (top-p) filtering |
| `_sample` | Sample a single token and return the full probability distribution |
| `_generate_draft_tokens` | Autoregressive draft generation with KV caching (gamma tokens) |
| `_verify_draft_tokens` | Single-pass target verification with rejection sampling |
| `trim_kv_cache` | Truncate a DynamicCache to discard rejected positions |
| `greedy_decode` | Baseline: `model.generate()` with greedy sampling |
| `greedy_decode_with_cache` | Baseline: manual token-by-token loop with explicit KV caching |


## Sampling modes

| Temperature | Behavior |
|---|---|
| `0.0` | Deterministic: accept iff `argmax(p_target) == draft_token` |
| `> 0.0` | Stochastic: accept with probability `min(p_target / p_draft, 1)`, rejection samples from the residual distribution |

Top-k and top-p filtering are applied identically to both draft and target probability distributions before the acceptance test, preserving the distributional guarantee.


## Models

| Role | Model | Parameters |
|---|---|---|
| Target | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B |
| Draft | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B |

The paper recommends the target be roughly 100x larger than the draft for good acceptance rates. The Qwen3 family shares the same tokenizer across all sizes, satisfying the shared-vocabulary requirement.

> [!IMPORTANT]
> Both models must share the same tokenizer vocabulary. The script asserts this at startup.


## Requirements

```
torch
transformers
```


## Usage

```python
from speculativeDecoding import SpeculativeDecoder, SpeculativeConfig

config = SpeculativeConfig(gamma=3, temperature=0.0, max_new_tokens=128)
decoder = SpeculativeDecoder(target_model, draft_model, tokenizer, config, device="cuda")

completion, metrics = decoder.decode(prompt)
print(completion)
print(f"Accepted {metrics['total_accepted']}/{metrics['total_proposed']} "
      f"({metrics['acceptance_ratio']:.1%})")
```

### Running the built-in benchmark

The script benchmarks three decoding strategies — greedy, greedy with KV cache, and speculative — and prints throughput and speedup:

```bash
python speculativeDecoding.py
```

## Benchmark harness

The `__main__` block runs all three strategies on the same prompt with `temperature=0.0` (deterministic) and reports:

| Metric | Description |
|---|---|
| Tokens generated | Total tokens produced |
| Wall time | End-to-end latency |
| Throughput | Tokens per second |
| Acceptance ratio | Fraction of draft tokens accepted by the target |
| Speedup | Relative to greedy and cached baselines |


## Credits

- Paper: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) — Leviathan, Kalman, Matias (2023)
- Models: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) and [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) by Alibaba Cloud
