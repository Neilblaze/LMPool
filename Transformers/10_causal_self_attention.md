# Lesson 10: Causal Self-Attention

Now we can compute attention.

Inputs:

```text
q.shape = [B, n_heads, T, head_dim]
k.shape = [B, n_heads, T, head_dim]
v.shape = [B, n_heads, T, head_dim]
```

Output:

```text
[B, n_heads, T, head_dim]
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Compute Attention Scores

Attention compares queries and keys:

```python
scores = q @ k.transpose(-2, -1)
```

Shape:

```text
[B, n_heads, T, head_dim]
@
[B, n_heads, head_dim, T]
->
[B, n_heads, T, T]
```

Each `[T, T]` matrix says:

```text
for each query position,
how much does it match each key position?
```

## Step 2: Scale The Scores

Use:

```python
scores = scores / math.sqrt(head_dim)
```

Why?

Dot products get larger when `head_dim` is larger. Scaling keeps the scores in a
healthier range before softmax.

## Step 3: Apply The Causal Mask

Decoder-only models cannot look ahead.

Create a lower-triangular mask:

```python
mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
```

For `T = 4`:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Allowed positions are `1`.

Future positions are `0`.

Apply it:

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

After this, softmax gives future positions probability `0`.

## Step 4: Softmax Into Attention Weights

```python
weights = torch.softmax(scores, dim=-1)
```

Shape:

```text
weights.shape = [B, n_heads, T, T]
```

Each row sums to `1`.

For each query position, weights say how much to read from each earlier key
position.

## Step 5: Read From Values

```python
out = weights @ v
```

Shape:

```text
[B, n_heads, T, T]
@
[B, n_heads, T, head_dim]
->
[B, n_heads, T, head_dim]
```

The output has one vector per position per head.

Each output vector is a weighted sum of value vectors.

## Step 6: Write The Attention Function

```python
import math
import torch
import torch.nn.functional as F


def causal_attention(q, k, v):
    B, n_heads, T, head_dim = q.shape

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = weights @ v
    return out
```

This is the core attention operation.

RoPE should be applied to `q` and `k` before this function.

## Step 7: Check The Shape

```python
q = torch.randn(2, 4, 8, 16)
k = torch.randn(2, 4, 8, 16)
v = torch.randn(2, 4, 8, 16)

out = causal_attention(q, k, v)
print(out.shape)
```

Expected:

```text
torch.Size([2, 4, 8, 16])
```

The output shape matches `q`, `k`, and `v`.

## Step 8: Understand Why This Is Self-Attention

It is self-attention because `q`, `k`, and `v` all come from the same `x`.

```text
x -> q
x -> k
x -> v
```

The sequence attends to itself.

It is causal because the mask blocks future positions.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain `q @ k.T`
- you can explain why scores are scaled
- you can build a lower-triangular causal mask
- you can explain why masked future scores become `-inf`
- you can trace `[B, heads, T, T] @ [B, heads, T, head_dim]`

Stop here. [Lesson 11](11_multi_head_attention.md) wraps this into multi-head attention.
