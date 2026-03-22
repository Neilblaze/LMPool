# miniLLM

A minimal, legible, and easy-to-modify decoder language model baseline. Built as a quick reference against which to test architecture and research ideas.

> [!NOTE]
> Based on [Mistral 7B](https://github.com/mistralai/mistral-src) with additions where necessary. Thanks to [Stella Biderman](https://twitter.com/BlancheMinerva/status/1740365334467756267) for aggregating best practices across recent LLMs.


## Architecture

- RMSNorm
- RoPE with interpolation / extrapolation (via [rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch))
- Flash Attention (via `F.scaled_dot_product_attention`)
- SwiGLU
- Pre-layer norms
- GQA / MQA
- KV caching
- MLA with compressed KV caching (DeepSeek V3)

**Model config:** no bias, no dropout. Use `4 * dim` for regular FF, `8/3 * dim` for GLU variants.


## Optimization

- AdamW, betas: 0.9, 0.95
- Gradient clipping: 1.0
- Cosine LR decay to 10% of peak LR
- LR warmup ~3% of total steps
- Weight decay: 0.1


## Training setup

- C4 dataset, streamed from Hugging Face (`datasets`)
- GPT-NeoX-20B tokenizer
- `torch.compile()`
- Automatic mixed precision (AMP)
- `pin_memory=True`, async data loading

Reference: [PyTorch performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)


## Some Notes

This is not built for inference speed, parallelism, or production use. It is a readable baseline for experimenting with architecture changes. The "state of the art" shifts constantly, so the goal is to make modifications fast and low-friction.

> [!IMPORTANT] 
> This configuration is not guaranteed to be optimal for every dataset or scale. There is valid disagreement about the value of individual components, but collectively they approximate a strong starting point. <br/>
> For more off-the-shelf transformer components, see [x-transformers](https://github.com/lucidrains/x-transformers) and [xformers](https://github.com/facebookresearch/xformers). They require more ramp-up time but offer more flexibility for production use.


## Open questions

- Sequence packing (T5-style, fitting multiple examples per context window)
- Pre-norm vs. post-norm tradeoffs at scale
- Warmup schedule sensitivity
- `optimizer.step` warning with GradScaler ([appears harmless](https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/7))

<br/>

> [!TIP]
> Might be useful when you want to replicate a benchmark run across multiple W&B projects.
> <br/>
> 
> ```python
> import wandb
> wandb.login()
> api = wandb.Api()
> 
> src_entity = "johnsmith"
> src_project = "project-1"
> src_name = "benchmark-run-1"
> 
> dst_entity = "johnsmith"
> dst_project = "project-2"
> 
> runs = api.runs(f"{src_entity}/{src_project}")
> 
> for run in runs:
>     if run.name == src_name:
>         history = run.history()
>         files = run.files()
> 
>         new_run = wandb.init(project=dst_project, entity=dst_entity, config=run.config, name=run.name, resume="allow")
> 
>         for index, row in history.iterrows():
>             step_size = history['_step'].values[1]
>             new_run.log(row.to_dict(), step=index * step_size)
> 
>         for file in files:
>             file.download(replace=True)
>             new_run.save(file.name, policy="now")
> 
>         new_run.finish()
> ```