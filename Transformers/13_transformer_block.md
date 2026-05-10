# Lesson 13: Transformer Block

Now combine:

```text
RMSNorm
multi-head causal attention with RoPE
SwiGLU MLP
residual connections
```

This is the core repeated unit of a modern decoder-only GPT.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Know The Block Formula

Use pre-norm:

```text
x = x + attention(rmsnorm_1(x))
x = x + mlp(rmsnorm_2(x))
```

The input and output shape is:

```text
[B, T, d_model]
```

The block updates the residual stream without changing its shape.

## Step 2: Define The Block

```python
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

That is the block.

Almost all the work is in the modules you already built.

## Step 3: Read The First Residual

```python
x = x + self.attn(self.norm1(x))
```

Step by step:

```text
norm1(x):
    stabilize vector scale

attn(...):
    mix information from earlier token positions

x + ...:
    add the attention update back to the stream
```

Shape:

```text
[B, T, d_model] -> [B, T, d_model]
```

## Step 4: Read The Second Residual

```python
x = x + self.mlp(self.norm2(x))
```

Step by step:

```text
norm2(x):
    stabilize vector scale again

mlp(...):
    transform each token position independently

x + ...:
    add the MLP update back to the stream
```

Shape stays:

```text
[B, T, d_model]
```

## Step 5: Check The Block

```python
block = TransformerBlock(config)
x = torch.randn(2, 4, 32)
out = block(x)

print(out.shape)
```

Expected:

```text
torch.Size([2, 4, 32])
```

This means the block can be stacked.

## Step 6: Stack Blocks

In PyTorch:

```python
blocks = nn.ModuleList([
    TransformerBlock(config)
    for _ in range(config.n_layers)
])
```

Forward:

```python
for block in blocks:
    x = block(x)
```

If:

```text
n_layers = 2
```

then:

```text
block 1 updates x
block 2 updates x again
```

The shape stays `[B, T, d_model]` the whole time.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can write the block formula from memory
- you know why it is called pre-norm
- you can explain both residual connections
- you know attention mixes positions and MLP transforms positions
- you can stack blocks because the shape stays unchanged

Stop here. [Lesson 14](14_tiny_gpt_model.md) builds the full GPT model.
