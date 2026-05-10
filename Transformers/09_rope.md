# Lesson 09: RoPE

RoPE means rotary position embeddings.

It gives attention position information by rotating query and key vectors.

In this course:

```text
RoPE applies to q and k
RoPE does not apply to v
RoPE does not get added to token embeddings
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Start With Q And K

From [Lesson 08](08_queries_keys_values.md):

```text
q.shape = [B, n_heads, T, head_dim]
k.shape = [B, n_heads, T, head_dim]
```

RoPE keeps the same shape:

```text
q_rot.shape = [B, n_heads, T, head_dim]
k_rot.shape = [B, n_heads, T, head_dim]
```

It changes how positions are represented inside the vectors.

## Step 2: Understand Rotation Pairs

RoPE treats nearby channels as pairs.

Example vector:

```text
[x0, x1, x2, x3, x4, x5, ...]
```

Pairs:

```text
(x0, x1)
(x2, x3)
(x4, x5)
```

Each pair gets rotated by an angle based on:

```text
position
channel pair
```

This is why `head_dim` should be even.

## Step 3: Build Frequencies

A common RoPE frequency setup:

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

Usually:

```text
theta = 10000
```

Shape:

```text
inv_freq.shape = [head_dim / 2]
```

For:

```text
head_dim = 8
```

we get:

```text
inv_freq.shape = [4]
```

## Step 4: Build Position Angles

Positions:

```python
positions = torch.arange(T)
```

Angles:

```python
freqs = torch.outer(positions.float(), inv_freq)
```

Shape:

```text
freqs.shape = [T, head_dim / 2]
```

Then:

```python
cos = freqs.cos()
sin = freqs.sin()
```

These are the rotation tables.

## Step 5: Rotate Half The Vector

For paired rotation, use:

```python
def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
```

If a pair is:

```text
(a, b)
```

`rotate_half` gives:

```text
(-b, a)
```

This is the piece needed for a 2D rotation.

## Step 6: Apply RoPE

RoPE formula:

```text
x_rot = x * cos + rotate_half(x) * sin
```

Code:

```python
def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
    return (x * cos) + (rotate_half(x) * sin)
```

The `None` dimensions make `cos` and `sin` broadcast over:

```text
batch
heads
```

The final shape stays:

```text
[B, n_heads, T, head_dim]
```

## Step 7: Create A RoPE Helper

```python
def build_rope_cache(T: int, head_dim: int, device, theta: float = 10000.0):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(T, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()
```

Usage:

```python
cos, sin = build_rope_cache(T, head_dim, device=q.device)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

Apply RoPE before computing attention scores.

## Step 8: Check The Shape

```python
q = torch.randn(2, 4, 8, 16)
cos, sin = build_rope_cache(T=8, head_dim=16, device=q.device)
q_rot = apply_rope(q, cos, sin)

print(q_rot.shape)
```

Expected:

```text
torch.Size([2, 4, 8, 16])
```

RoPE changed the values, not the shape.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you know RoPE applies to q and k
- you know RoPE rotates channel pairs
- you know `head_dim` should be even
- you can build `cos` and `sin` tables
- you can explain that RoPE preserves shape

Stop here. [Lesson 10](10_causal_self_attention.md) builds causal self-attention.
