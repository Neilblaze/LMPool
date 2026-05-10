# Lesson 08: Queries, Keys, And Values

Attention starts by turning the residual stream into three tensors:

```text
q = queries
k = keys
v = values
```

This lesson does not compute attention scores yet.

It only teaches where `q`, `k`, and `v` come from and how their shapes work.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Start With The Residual Stream

Input to attention:

```text
x.shape = [B, T, d_model]
```

Example config:

```text
B = 2
T = 4
d_model = 32
n_heads = 4
head_dim = 8
```

Attention splits the model dimension into heads:

```text
d_model = n_heads * head_dim
32 = 4 * 8
```

## Step 2: Create Q, K, V Projections

Each token vector is projected into query, key, and value vectors.

In code:

```python
from torch import nn


q_proj = nn.Linear(d_model, d_model, bias=False)
k_proj = nn.Linear(d_model, d_model, bias=False)
v_proj = nn.Linear(d_model, d_model, bias=False)
```

Each projection keeps the last dimension at `d_model`:

```text
[B, T, d_model] -> [B, T, d_model]
```

So:

```text
q.shape = [B, T, d_model]
k.shape = [B, T, d_model]
v.shape = [B, T, d_model]
```

## Step 3: Understand The Roles

Use this mental model:

```text
query:
    what this position is looking for

key:
    what this position offers to be matched against

value:
    what information this position will pass along if attended to
```

Attention compares queries with keys to decide how much to read from values.

```text
q and k decide the weights
v carries the content
```

## Step 4: Split Into Heads

Before heads:

```text
q.shape = [B, T, d_model]
```

After heads:

```text
q.shape = [B, n_heads, T, head_dim]
```

Code:

```python
B, T, C = q.shape

q = q.reshape(B, T, n_heads, head_dim)
q = q.transpose(1, 2)
```

The first line reshapes:

```text
[B, T, d_model] -> [B, T, n_heads, head_dim]
```

The transpose moves heads before time:

```text
[B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
```

Do the same for `k` and `v`.

## Step 5: Write A Helper

```python
def split_heads(x: torch.Tensor, n_heads: int):
    B, T, C = x.shape
    head_dim = C // n_heads
    x = x.reshape(B, T, n_heads, head_dim)
    return x.transpose(1, 2)
```

Input:

```text
[B, T, d_model]
```

Output:

```text
[B, n_heads, T, head_dim]
```

This layout makes attention math easier.

## Step 6: Combine Heads Later

After attention, heads must be merged back:

```text
[B, n_heads, T, head_dim] -> [B, T, d_model]
```

Helper:

```python
def combine_heads(x: torch.Tensor):
    B, n_heads, T, head_dim = x.shape
    x = x.transpose(1, 2)
    return x.reshape(B, T, n_heads * head_dim)
```

This reverses `split_heads`.

## Step 7: Check The Shapes

```python
x = torch.randn(2, 4, 32)

q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

q = split_heads(q, n_heads=4)
k = split_heads(k, n_heads=4)
v = split_heads(v, n_heads=4)

print(q.shape, k.shape, v.shape)
```

Expected:

```text
torch.Size([2, 4, 4, 8])
torch.Size([2, 4, 4, 8])
torch.Size([2, 4, 4, 8])
```

This means:

```text
2 batch rows
4 heads
4 positions
8 numbers per head
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain what q, k, and v mean
- you can compute `head_dim`
- you can split `[B, T, d_model]` into `[B, n_heads, T, head_dim]`
- you can combine heads back into `[B, T, d_model]`
- you know RoPE will apply to q and k next

Stop here. [Lesson 09](09_rope.md) builds RoPE.
