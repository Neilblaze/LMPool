# Lesson 07: RMSNorm

Modern GPT-style models usually normalize the residual stream before attention
and before the MLP.

In this course, we use RMSNorm.

The transformer block will look like:

```text
x = x + attention(rmsnorm(x))
x = x + mlp(rmsnorm(x))
```

This lesson builds RMSNorm.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Know Why We Normalize

The residual stream has shape:

```text
x.shape = [B, T, d_model]
```

Every token position has a vector of size `d_model`.

During training, vector scales can drift. If one layer outputs vectors that are
too large or too small, the next layer has a harder job.

RMSNorm keeps vector scale more stable.

It does not mix token positions.

It normalizes each position's vector independently.

## Step 2: Understand RMS

RMS means root mean square.

For one vector:

```text
x = [x1, x2, x3, ...]
```

RMS is:

```text
sqrt(mean(x^2))
```

If a vector has a large RMS, divide by that value to bring the scale down.

If a vector has a tiny RMS, division would be unstable, so we add a small
epsilon.

## Step 3: RMSNorm Formula

For each token vector:

```text
rms = sqrt(mean(x * x) + eps)
normalized = x / rms
output = normalized * weight
```

`weight` is learned.

Shape:

```text
weight.shape = [d_model]
```

It lets the model choose the final scale for each channel.

## Step 4: Implement RMSNorm

```python
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight
```

The important dimension is:

```python
dim=-1
```

That means normalize across the channel dimension.

For:

```text
x.shape = [B, T, d_model]
```

RMS is computed separately for each `[B, T]` position.

## Step 5: Check The Shape

```python
norm = RMSNorm(d_model=32)
x = torch.randn(2, 4, 32)
y = norm(x)

print(y.shape)
```

Expected:

```text
torch.Size([2, 4, 32])
```

RMSNorm changes values, not shape.

## Step 6: Place RMSNorm In The Block

Use pre-norm:

```python
x = x + attention(rmsnorm_1(x))
x = x + mlp(rmsnorm_2(x))
```

Why two RMSNorm layers?

Because attention and MLP each get their own normalized input.

The residual additions happen after the sublayer returns.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain RMS as `sqrt(mean(x^2))`
- you can explain why RMSNorm uses `dim=-1`
- you can implement `RMSNorm`
- you know RMSNorm keeps `[B, T, d_model]` unchanged
- you know RMSNorm is used before attention and before MLP

Stop here. [Lesson 08](08_queries_keys_values.md) builds queries, keys, and values.
