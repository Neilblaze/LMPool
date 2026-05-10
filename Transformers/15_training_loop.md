# Lesson 15: Training Loop

Training repeatedly does:

```text
sample batch
forward pass
compute loss
backward pass
optimizer step
repeat
```

This lesson builds that loop.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Create The Model And Optimizer

```python
config = GPTConfig()
model = TinyGPT(config)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)
```

AdamW is the standard starting optimizer for transformer training.

The learning rate is intentionally conservative.

## Step 2: Sample A Batch

From [Lesson 02](02_training_batches.md):

```python
x, y = get_batch(
    data=train_data,
    batch_size=32,
    context_length=config.context_length,
)
```

Shapes:

```text
x.shape = [B, T]
y.shape = [B, T]
```

The batch creates the next-token prediction task.

## Step 3: Forward Pass

```python
logits, loss = model(x, y)
```

The model returns:

```text
logits:
    [B, T, vocab_size]

loss:
    scalar tensor
```

The loss is the number we want to reduce.

## Step 4: Backward Pass

Before backprop:

```python
optimizer.zero_grad(set_to_none=True)
```

Then:

```python
loss.backward()
```

This computes gradients for model parameters.

Then:

```python
optimizer.step()
```

This updates the weights.

## Step 5: Full Training Loop

```python
for step in range(1000):
    x, y = get_batch(train_data, batch_size=32, context_length=config.context_length)

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())
```

This is the core loop.

Everything else is improvement around it.

## Step 6: Move To Device

Use GPU when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
train_data = train_data.to(device)
```

Then `get_batch` will return tensors on the same device if `data` is already on
that device.

The model and batch must be on the same device.

## Step 7: Watch The Loss

At the start, loss is often near:

```text
log(vocab_size)
```

During training, it should trend downward.

It will not decrease smoothly every step.

What matters is the trend over many steps.

If loss is `nan`, common causes are:

```text
learning rate too high
bad mask
bad shape
invalid targets
```

## Step 8: Add Gradient Clipping

For stability:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Place it after `loss.backward()` and before `optimizer.step()`:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

This prevents very large gradients from creating a wild update.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can write the training loop from memory
- you know why `zero_grad` comes before `backward`
- you know `backward` computes gradients
- you know `optimizer.step` updates weights
- you can watch loss and explain the trend

Stop here. [Lesson 16](16_validation_and_sampling.md) adds validation and sampling.
