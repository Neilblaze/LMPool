# Lesson 12: SwiGLU MLP

Every transformer block has attention and an MLP.

Attention mixes information across token positions.

The MLP transforms each position independently.

Modern LLMs often use a gated MLP such as SwiGLU.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Know The Shape Contract

Input:

```text
x.shape = [B, T, d_model]
```

Output:

```text
out.shape = [B, T, d_model]
```

The MLP may expand the hidden dimension internally, but it returns to `d_model`
so it can be added back to the residual stream.

## Step 2: Understand The Gate

SwiGLU uses two projections:

```text
gate = gate_proj(x)
up = up_proj(x)
```

Then:

```text
hidden = silu(gate) * up
```

The gate decides what information passes through.

Then a down projection returns to `d_model`:

```text
out = down_proj(hidden)
```

## Step 3: Choose Hidden Size

Use:

```text
hidden_dim = hidden_multiplier * d_model
```

With:

```text
d_model = 32
hidden_multiplier = 4
```

we get:

```text
hidden_dim = 128
```

Shape flow:

```text
[B, T, 32]
-> [B, T, 128]
-> [B, T, 32]
```

## Step 4: Implement SwiGLU

```python
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = config.hidden_multiplier * config.d_model
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)
```

Important line:

```python
hidden = F.silu(gate) * up
```

That is the gated part.

`silu` is also called swish:

```text
silu(x) = x * sigmoid(x)
```

## Step 5: Check The Shape

```python
mlp = SwiGLU(config)
x = torch.randn(2, 4, 32)
out = mlp(x)

print(out.shape)
```

Expected:

```text
torch.Size([2, 4, 32])
```

The MLP changes values, not the outer shape.

## Step 6: Place It In The Block

The block will use:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

The MLP receives normalized vectors, transforms each position, and returns a
tensor that can be added back to `x`.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain the gate/up/down projections
- you can explain why `hidden_dim` is larger than `d_model`
- you can implement `SwiGLU`
- you know the MLP returns `[B, T, d_model]`

Stop here. [Lesson 13](13_transformer_block.md) builds the full transformer block.
