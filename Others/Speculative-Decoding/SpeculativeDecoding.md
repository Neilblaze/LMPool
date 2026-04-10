# Faster LLM Inference with Speculative Decoding

Speculative Decoding is a latency reduction technique for large language model (LLM) inference.

Instead of letting the big model (the target model) generate each token sequentially, we first use a smaller, faster draft model to propose multiple candidate tokens ahead of time. Then the large model verifies these proposals in parallel, significantly reducing the number of expensive forward passes.

## Background
Inference from large autoregressive models like Transformers is slow — decoding $K$ tokens takes $K$ serial runs of the model.

**Key observations:**
- **Variable Step Difficulty**: Some inference steps are "harder" (require target model precision) while others are "easier" and can be accurately predicted by smaller models.
<p align="center">
  <img src="https://res.cloudinary.com/dmlwye965/image/upload/v1775849045/speculative_decoding_easy_hard_ynoizi.webp" alt="Easy vs hard decoding steps in speculative decoding" width="80%">
  <br>
  <em>Image source: <a href="https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120">Speculative Decoding — Make LLM Inference Faster</a></em>
</p>

- **Memory Bound vs Compute Bound**: Large model inference is often not bottlenecked by arithmetic operations, but rather by memory bandwidth and communication (loading weights from HBM to SRAM). Small models can generate tokens much faster as they are exponentially lighter on memory I/O.

## Speculative Decoding
Let $M_p$ be the target model and $M_q$ be a more efficient draft model. The core idea is to:

1. Use $M_q$ to autoregressively generate $\gamma \in \mathbb{Z}^+$ completions.
2. Use $M_p$ to evaluate all candidate tokens and their respective probabilities from $M_q$ in parallel (single forward pass).
3. Sample an additional token from an adjusted distribution to fix the first rejected match, or add a bonus token if all are accepted.

This way, each parallel run of $M_p$ produces at least one token, but can generate up to $\gamma + 1$ tokens depending on how well $M_q$ approximates $M_p$.

## Speculative Sampling
To sample $x \sim p(x)$, we instead sample $x \sim q(x)$, keeping it with probability $\min(1, p(x)/q(x))$. If rejected, we resample from $p'(x) = \text{norm}(\max(0, p(x) - q(x)))$.

It can be proven (see Appendix A.1 of [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)) that $x$ sampled this way follows the exact distribution of $p(x)$.

<p align="center">
  <img src="https://res.cloudinary.com/dmlwye965/image/upload/v1775848403/speculative_decoding_algo_cqcq8c.png" alt="Speculative Decoding Algorithm" width="90%">
</p>

## References
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Speculative Decoding — Make LLM Inference Faster](https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120)