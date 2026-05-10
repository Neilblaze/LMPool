# Lesson 03: Model Config And Shape Flow

[Lesson 02](02_training_batches.md) created training batches:

```text
x.shape = [B, T]
y.shape = [B, T]
```

This lesson defines the model configuration for a small modern GPT and traces
the shapes through the whole forward pass.

No attention math yet.

No RoPE implementation yet.

Just the config and the tensor flow we are going to build.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Define The Config

A GPT model needs a small set of architecture numbers.

Use this tiny config:

```python
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 6
    context_length: int = 4
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    hidden_multiplier: int = 4
```

Read it as:

```text
vocab_size:
    how many token ids exist

context_length:
    how many positions the model sees

d_model:
    size of each token vector

n_layers:
    number of transformer blocks

n_heads:
    number of attention heads per block

hidden_multiplier:
    controls the MLP hidden size
```

This is small on purpose. The architecture is modern, but the numbers are toy
numbers so every shape is easy to follow.

## Step 2: Check The Attention Head Size

Attention splits `d_model` across heads.

With:

```text
d_model = 32
n_heads = 4
```

each head gets:

```text
head_dim = d_model / n_heads = 8
```

In code:

```python
head_dim = config.d_model // config.n_heads
```

This division must be exact.

If `d_model = 30` and `n_heads = 4`, the heads would not split evenly. Avoid
that.

Useful check:

```python
assert config.d_model % config.n_heads == 0
```

## Step 3: Start With Token IDs

From [Lesson 02](02_training_batches.md):

```text
x.shape = [B, T]
```

Example:

```text
B = 2
T = 4
```

So:

```text
x.shape = [2, 4]
```

Every value inside `x` is an integer token id:

```text
[
  [1, 2, 3, 4],
  [5, 1, 2, 3],
]
```

The model cannot run attention over integer ids directly. It first needs token
vectors.

## Step 4: Token Embedding Shape

The token embedding table has shape:

```text
[vocab_size, d_model]
```

With our config:

```text
[6, 32]
```

That means:

```text
6 possible token ids
32 numbers per token vector
```

When we look up embeddings for `x`:

```text
[B, T] -> [B, T, d_model]
```

With numbers:

```text
[2, 4] -> [2, 4, 32]
```

This tensor is the residual stream.

Call it:

```text
h.shape = [B, T, d_model]
```

## Step 5: Transformer Blocks Keep The Main Shape

Each transformer block receives:

```text
h.shape = [B, T, d_model]
```

And returns:

```text
h.shape = [B, T, d_model]
```

The content changes, but the outer shape stays the same.

For our tiny config:

```text
before block 1: [2, 4, 32]
after block 1:  [2, 4, 32]
after block 2:  [2, 4, 32]
```

This is why transformer blocks can be stacked.

The output of one block is the input to the next.

## Step 6: Q, K, V Shapes Inside Attention

Inside attention, the model creates queries, keys, and values:

```text
q, k, v
```

Before splitting heads, each has shape:

```text
[B, T, d_model]
```

After splitting into heads:

```text
[B, n_heads, T, head_dim]
```

With our config:

```text
[2, 4, 4, 8]
```

Read that carefully:

```text
2 batch rows
4 attention heads
4 time positions
8 numbers per head
```

RoPE will apply to `q` and `k` after they are shaped for attention.

The shape stays the same:

```text
q with RoPE: [B, n_heads, T, head_dim]
k with RoPE: [B, n_heads, T, head_dim]
```

## Step 7: Attention Output Shape

Attention computes how each position reads earlier positions.

After attention, the heads are combined back together:

```text
[B, n_heads, T, head_dim] -> [B, T, d_model]
```

With numbers:

```text
[2, 4, 4, 8] -> [2, 4, 32]
```

The block adds this result back to the residual stream:

```text
h = h + attention_output
```

The shape stays:

```text
[2, 4, 32]
```

## Step 8: SwiGLU MLP Shape

The MLP works on each position independently.

It usually expands the hidden size, applies a gated activation, then projects
back down.

Using:

```text
hidden_multiplier = 4
d_model = 32
```

the MLP hidden size is:

```text
mlp_hidden = 4 * 32 = 128
```

The shape flow is:

```text
[B, T, d_model]
-> [B, T, mlp_hidden]
-> [B, T, d_model]
```

With numbers:

```text
[2, 4, 32]
-> [2, 4, 128]
-> [2, 4, 32]
```

Then the block adds the MLP output back:

```text
h = h + mlp_output
```

Again:

```text
h.shape = [B, T, d_model]
```

## Step 9: Final Norm And LM Head

After all transformer blocks, the model applies final RMSNorm:

```text
[B, T, d_model] -> [B, T, d_model]
```

Then the LM head maps hidden vectors to vocabulary logits:

```text
[B, T, d_model] -> [B, T, vocab_size]
```

With our config:

```text
[2, 4, 32] -> [2, 4, 6]
```

For every batch row and every time position, the model outputs one score per
token in the vocabulary.

That is what the loss compares to `y`.

## Step 10: Know The Full Shape Flow

Here is the complete forward pass:

```text
x token ids:
    [B, T]

token embedding:
    [B, T] -> [B, T, d_model]

for each transformer block:
    RMSNorm keeps [B, T, d_model]
    attention uses q/k/v [B, n_heads, T, head_dim]
    RoPE applies to q/k
    attention returns [B, T, d_model]
    SwiGLU returns [B, T, d_model]

final RMSNorm:
    [B, T, d_model]

LM head:
    [B, T, d_model] -> [B, T, vocab_size]
```

With the tiny config:

```text
[2, 4]
-> [2, 4, 32]
-> [2, 4, 32]
-> [2, 4, 6]
```

## Step 11: Write Your Lesson 03 Notes

Write:

```markdown
# Lesson 03 Notes

## Config

vocab_size =
context_length =
d_model =
n_layers =
n_heads =
head_dim =

## Main Shape Flow

x:
token embeddings:
after blocks:
logits:

## Attention Shapes

q/k/v before heads:
q/k/v after heads:
where RoPE applies:

## MLP Shape

MLP expands from:
MLP projects back to:
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain each config field
- you can compute `head_dim`
- you can trace `[B, T] -> [B, T, d_model] -> [B, T, vocab_size]`
- you know RoPE applies to `q` and `k`
- you know transformer blocks keep the main `[B, T, d_model]` shape

Stop here. [Lesson 04](04_token_embeddings_and_lm_head.md) starts coding token embeddings and the LM head.
