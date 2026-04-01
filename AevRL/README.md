# AevRL

A lightweight _Reinforcement Learning_ stack for training language models with [GRPO](https://abderrahmanskiredj.github.io/the-illustrated-grpo) (Group Relative Policy Optimization). The main training loop is under 500 lines of code. Built to be hackable, modular, and straightforward to extend with new algorithms and environments.

AevRL runs RL rollouts against a chat model served by vLLM, collects rewards from a pluggable environment, and trains a local LoRA adapter using clipped policy gradients with a KL penalty against the frozen base model.


## How It Works

Each training step follows this sequence:

1. Wake the vLLM server and load the latest LoRA adapter
2. Run `n_rollouts` async rollouts through the configured environment, collecting conversation transcripts and rewards
3. Tokenize each rollout and query vLLM for per-token log-probabilities from both the adapter model (old policy) and the base model (reference)
4. Sleep the vLLM server to reclaim GPU memory
5. Move the trainable adapter and optimizer onto GPU
6. Compute group-normalized advantages across rollouts sharing the same `group_id`
7. Run `n_iters` gradient accumulation passes of the GRPO loss (clipped surrogate + KL penalty)
8. Save the updated adapter, offload to CPU, and repeat

The vLLM server handles inference while PyTorch handles training. The `--enable-sleep-mode` flag allows them to time-share a single GPU.


## Algorithm

AevRL ships with GRPO as the default algorithm. The implementation follows the grouped rollout formulation:

- **Advantage estimation** groups rollouts by `group_id` and normalizes rewards within each group. This eliminates the need for a value function.
- **Policy loss** uses a clipped importance-weighted objective, identical to PPO-clip, applied per-token with assistant-turn masking.
- **KL penalty** uses the unbiased estimator `exp(r) - r - 1` where `r = log(pi/pi_ref)`, scaled by `kl_coef`.

The algorithm interface (`src/algo/base.py`) requires only two methods, `compute_advantages` and `loss`, which makes it straightforward to implement alternatives like REINFORCE, RLOO, or standard PPO with a learned value function.

<br/>

![NOTE](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

<br/>


## Environments

An environment defines task generation and reward computation. It does not touch training objectives or optimizer logic.

The `Environment` ABC (`src/rl/env.py`) requires three things:

- `system_prompt` (property) that defines the model's role
- `next_query` (property) that returns the next user message
- `step(response)` that evaluates the model's output, sets `self.reward`, and marks `self.done`

An `EnvironmentFactory` creates environment instances for each rollout. The factory is loaded dynamically from the config path:

```yaml
env:
  factory: environments.gsm8k:GSM8KEnvironmentFactory
  kwargs: {}
```

### Included environments

🟠 **SimpleMath** generates basic arithmetic problems (addition, subtraction, multiplication of small integers). Rewards structured output format (+0.1 each for `<think>` and `<answer>` blocks) and correctness (+1.0 for the right answer).

🟠 **GSM8K** streams examples from the [OpenAI GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k) with thread-safe shuffled iteration. Uses the same `<think>/<answer>` format rewards as SimpleMath, plus a +0.05 bonus for parseable numeric answers. Reference answers are extracted from the `####` delimiter and normalized (comma/dollar stripping, decimal canonicalization).

> [!TIP]
> Both environments emit `group_id` in their metadata. This is required by GRPO for within-group advantage normalization. Any custom environment intended to work with GRPO must also set `group_id` in `self._metadata`.


<br/>

![NOTE](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

<br/>

## Configuration

All configuration is defined in a single YAML file, validated by Pydantic models in `src/utils/config.py`.

| Section | Key fields | Purpose |
|---|---|---|
| `train` | `model_name`, `adapter_path`, `lr`, `n_steps`, `n_iters`, `train_microbatch_size` | Base model, adapter save path, optimizer and gradient accumulation settings |
| `rollout` | `n_rollouts`, `max_parallel_rollouts`, `rollout_timeout` | Rollout parallelism and timeouts |
| `algo` | `factory`, `kwargs` (n_groups, clip_eps, kl_coef, group_adv_eps) | Algorithm class and its hyperparameters |
| `env` | `factory`, `kwargs` | Environment factory import path and constructor arguments |

> [!IMPORTANT]
> `rollout.n_rollouts` must be evenly divisible by `algo.kwargs.n_groups`. Each group receives `n_rollouts / n_groups` rollouts, all sharing the same problem.

[FlashAttention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention) is auto-detected. If the `flash_attn` package is installed, `train.use_flash_attn` defaults to `True` and the model uses `attn_implementation="flash_attention_2"`.

Assistant masking uses Qwen-style chat tokens (`<|im_start|>assistant` / `<|im_end|>`) by default. The token IDs are resolved automatically from the tokenizer at config load time.


## Requirements

- Python 3.11+
- CUDA-capable GPU
- [uv](https://docs.astral.sh/uv/) package manager
- vLLM (installed separately)


## Setup

Install base dependencies:

```bash
uv sync
```

Install FlashAttention support (optional, Linux only):

```bash
uv sync --extra flash_attn --no-build-isolation
```

Install vLLM separately (it is used as an external runtime and is not declared in `pyproject.toml`):

```bash
uv pip install vllm
```

If using the GSM8K environment, also install `datasets`:

```bash
uv pip install datasets
```

---

## Running

### Start vLLM

The trainer expects a local vLLM server at `http://localhost:8000` (override with `VLLM_BASE_URL`).

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_SERVER_DEV_MODE=1 \
vllm serve Qwen/Qwen3.5-2B \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --enable-sleep-mode \
  --enable-lora
```

| Flag | Why |
|---|---|
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` | Enables runtime adapter load/unload via the REST API |
| `VLLM_SERVER_DEV_MODE=1` | Enables the nonstandard `/sleep` and `/wake_up` endpoints used by the trainer |
| `--enable-sleep-mode` | Allows the server to release GPU memory so training can use it |
| `--enable-lora` | Required because the trainer hot-reloads the LoRA adapter at every training step |

### Start training

```bash
# Simple math
python -m src.rl.train --config configs/simple_math.yaml

# GSM8K
python -m src.rl.train --config configs/gsm8k.yaml
```

> [!NOTE]
> The LoRA adapter directory is created automatically on the first run if it does not already exist. Subsequent runs resume from the saved adapter.


## Notes

- The default configs use `Qwen/Qwen3.5-2B` as the base model. Adjust `model_name` and the vLLM serve command to match your model.
- LoRA targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` with rank 16 and alpha 32. These are currently hardcoded in `train.py`.
- Gradient checkpointing is enabled by default to reduce memory usage during training.
- W&B logging is enabled by default. Set `use_wandb: false` in the config to disable it.
- The trainer offloads the model and optimizer to CPU between training steps to share GPU memory with vLLM.
