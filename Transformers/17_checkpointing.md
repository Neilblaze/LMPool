# Lesson 17: Checkpointing

Training can take time.

You need to save progress.

A checkpoint should store:

```text
model weights
optimizer state
model config
current step
latest losses
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Save A Checkpoint

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": config,
    "step": step,
    "losses": losses,
}

torch.save(checkpoint, "checkpoint.pt")
```

`model.state_dict()` stores learned weights.

`optimizer.state_dict()` stores optimizer momentum and related state.

The config tells you how to rebuild the model shape.

## Step 2: Load A Checkpoint

```python
checkpoint = torch.load("checkpoint.pt", map_location=device)

config = checkpoint["config"]
model = TinyGPT(config).to(device)
model.load_state_dict(checkpoint["model"])

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.load_state_dict(checkpoint["optimizer"])

step = checkpoint["step"]
```

Order matters:

```text
create model with same config
load model weights
create optimizer
load optimizer state
```

## Step 3: Save Only For Inference

For generation, you do not need optimizer state.

```python
torch.save(
    {
        "model": model.state_dict(),
        "config": config,
    },
    "model.pt",
)
```

This file is smaller.

Use training checkpoints while training.

Use inference checkpoints when sharing a trained model.

## Step 4: Save Periodically

Inside training:

```python
if step % save_interval == 0:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "losses": losses,
        },
        "checkpoint.pt",
    )
```

Do not wait until the end of training.

Crashes happen.

## Step 5: Verify A Loaded Model

After loading:

```python
model.eval()
prompt = torch.tensor([[1]], dtype=torch.long, device=device)
out = model.generate(prompt, max_new_tokens=20)
print(out)
```

If loading works, the model should generate without shape errors.

For exact reproducibility, you also need random seeds and deterministic settings,
but checkpointing weights is the essential part.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can save model weights
- you can save optimizer state
- you can reload the model from config
- you know the difference between training and inference checkpoints
- you can generate after loading

Stop here. [Lesson 18](18_kv_cache.md) explains KV cache for faster generation.
