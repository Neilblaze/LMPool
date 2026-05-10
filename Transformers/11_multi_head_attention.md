# Lesson 11: Multi-Head Attention With RoPE

This lesson wraps the previous pieces into one module.

The module will:

```text
x -> q, k, v
split heads
apply RoPE to q and k
run causal attention
combine heads
project output
```

Input:

```text
x.shape = [B, T, d_model]
```

Output:

```text
out.shape = [B, T, d_model]
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Define The Module Fields

```python
import math
import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
```

The first three projections create attention inputs.

The output projection mixes information after heads are combined.

## Step 2: Split And Combine Heads

Inside the class:

```python
def split_heads(self, x):
    B, T, C = x.shape
    x = x.reshape(B, T, self.n_heads, self.head_dim)
    return x.transpose(1, 2)

def combine_heads(self, x):
    B, n_heads, T, head_dim = x.shape
    x = x.transpose(1, 2)
    return x.reshape(B, T, n_heads * head_dim)
```

Split:

```text
[B, T, d_model] -> [B, n_heads, T, head_dim]
```

Combine:

```text
[B, n_heads, T, head_dim] -> [B, T, d_model]
```

## Step 3: Include RoPE Helpers

```python
def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def build_rope_cache(T, head_dim, device, theta=10000.0):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(T, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
    return (x * cos) + (rotate_half(x) * sin)
```

RoPE applies after splitting heads because it works per head.

## Step 4: Write The Forward Pass

```python
def forward(self, x):
    B, T, C = x.shape

    q = self.split_heads(self.q_proj(x))
    k = self.split_heads(self.k_proj(x))
    v = self.split_heads(self.v_proj(x))

    cos, sin = build_rope_cache(T, self.head_dim, device=x.device)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(self.head_dim)

    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = weights @ v

    out = self.combine_heads(out)
    out = self.out_proj(out)
    return out
```

This is the complete attention module.

## Step 5: Trace The Shapes

For:

```text
B = 2
T = 4
d_model = 32
n_heads = 4
head_dim = 8
```

Shape flow:

```text
x:          [2, 4, 32]
q/k/v:      [2, 4, 32]
split:      [2, 4, 4, 8]
RoPE:       [2, 4, 4, 8]
scores:     [2, 4, 4, 4]
attn out:   [2, 4, 4, 8]
combined:   [2, 4, 32]
projected:  [2, 4, 32]
```

The module starts and ends with `[B, T, d_model]`.

That is why it fits inside the residual block.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can trace the full attention module
- you know RoPE applies after splitting heads
- you can explain why attention scores are `[B, heads, T, T]`
- you can explain why the output returns to `[B, T, d_model]`

Stop here. [Lesson 12](12_swiglu_mlp.md) builds the SwiGLU MLP.
