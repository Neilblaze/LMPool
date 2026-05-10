# Lesson 16: Validation And Sampling

Training loss tells you how the model is doing on batches it trains on.

Validation loss checks data held out from training.

Sampling shows what the model produces.

You need both.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Split Data

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

Training data updates weights.

Validation data does not.

## Step 2: Estimate Loss Without Gradients

```python
@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, eval_iters=20):
    model.eval()
    out = {}

    for split, split_data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split_data, batch_size, context_length)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out
```

`model.eval()` changes behavior for layers like dropout.

Even if your first model has no dropout, use the habit.

`torch.no_grad()` saves memory because validation does not update weights.

## Step 3: Print Loss During Training

```python
for step in range(max_steps):
    if step % eval_interval == 0:
        losses = estimate_loss(
            model,
            train_data,
            val_data,
            batch_size=32,
            context_length=config.context_length,
        )
        print(step, losses)

    x, y = get_batch(train_data, 32, config.context_length)
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Now you can compare:

```text
train loss
val loss
```

If train loss drops but val loss gets worse, the model may be overfitting.

## Step 4: Sample Text

Start from a prompt token:

```python
prompt = torch.tensor([[1]], dtype=torch.long, device=device)
out = model.generate(prompt, max_new_tokens=50)
```

Then decode ids back to text with your tokenizer:

```python
print(decode(out[0].tolist()))
```

At first, samples will be messy.

That is normal.

Sampling is a qualitative check, not a replacement for validation loss.

## Step 5: Use Temperature

Temperature controls randomness:

```python
next_logits = next_logits / temperature
```

Lower temperature:

```text
more conservative
```

Higher temperature:

```text
more random
```

A simple generate signature:

```python
def generate(self, input_ids, max_new_tokens, temperature=1.0):
```

Then:

```python
next_logits = logits[:, -1, :] / temperature
```

Keep `temperature > 0`.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can split train and validation data
- you can estimate loss without gradients
- you can explain `model.train()` and `model.eval()`
- you can sample generated tokens
- you can explain temperature

Stop here. [Lesson 17](17_checkpointing.md) saves and loads checkpoints.
