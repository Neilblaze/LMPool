# Lesson 04: Token Embeddings And The LM Head

[Lesson 03](03_model_config_and_shapes.md) defined the shape flow:

```text
[B, T] -> [B, T, d_model] -> [B, T, vocab_size]
```

This lesson builds the first model shell that follows that contract.

It will not be powerful yet. It has no attention, no RoPE, no RMSNorm, and no
SwiGLU block yet.

That is intentional.

The goal is to build the outer frame of a GPT model:

```text
token ids
-> token embeddings
-> transformer blocks later
-> final norm later
-> LM head
-> logits
```

Once this shell works, every future lesson fills in one real modern GPT piece.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Start From The Config

Use the same config shape from [Lesson 03](03_model_config_and_shapes.md):

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

For this lesson, the most important fields are:

```text
vocab_size:
    how many possible token ids exist

d_model:
    how wide each token vector is
```

With the tiny config:

```text
vocab_size = 6
d_model = 32
```

The model will turn each token id into a vector of size `32`, then turn each
vector back into `6` vocabulary scores.

## Step 2: Understand Token Embedding

The input batch contains token ids:

```text
x.shape = [B, T]
```

Example:

```text
x = [
  [1, 2, 3, 4],
  [5, 1, 2, 3],
]
```

Each number is an index into the vocabulary.

An embedding table is a learned lookup table:

```text
token_embedding.shape = [vocab_size, d_model]
```

With our config:

```text
token_embedding.shape = [6, 32]
```

That means:

```text
token id 0 has a learned vector of size 32
token id 1 has a learned vector of size 32
...
token id 5 has a learned vector of size 32
```

When you pass `x` through the embedding table:

```text
[B, T] -> [B, T, d_model]
```

With numbers:

```text
[2, 4] -> [2, 4, 32]
```

This output is the beginning of the residual stream.

## Step 3: Write The Embedding Layer

In PyTorch:

```python
import torch
from torch import nn


token_embedding = nn.Embedding(
    num_embeddings=6,
    embedding_dim=32,
)
```

The names mean:

```text
num_embeddings:
    number of rows in the table, usually vocab_size

embedding_dim:
    number of columns in the table, usually d_model
```

If:

```python
x = torch.tensor([
    [1, 2, 3, 4],
    [5, 1, 2, 3],
], dtype=torch.long)
```

then:

```python
h = token_embedding(x)
print(h.shape)
```

should produce:

```text
torch.Size([2, 4, 32])
```

The integer ids disappeared. The model now has one learned vector per token
position.

## Step 4: Understand The LM Head

At the end of a GPT model, each hidden vector must become vocabulary scores.

The LM head is a linear layer:

```text
[d_model] -> [vocab_size]
```

For every position:

```text
hidden vector of size 32 -> 6 logits
```

In PyTorch:

```python
lm_head = nn.Linear(32, 6, bias=False)
```

If:

```text
h.shape = [2, 4, 32]
```

then:

```python
logits = lm_head(h)
```

produces:

```text
logits.shape = [2, 4, 6]
```

Each `logits[b, t]` vector contains one score for every token id in the
vocabulary.

Example:

```text
logits[0, 2] = scores for the next token after row 0, position 2
```

## Step 5: Create A Small Model Shell

Now wrap the embedding and LM head in a module:

```python
import torch
from torch import nn


class TinyGPTShell(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        h = self.token_embedding(input_ids)
        logits = self.lm_head(h)
        return logits
```

Read the forward pass slowly:

```python
h = self.token_embedding(input_ids)
```

This changes:

```text
[B, T] -> [B, T, d_model]
```

Then:

```python
logits = self.lm_head(h)
```

This changes:

```text
[B, T, d_model] -> [B, T, vocab_size]
```

There are no transformer blocks yet. This is the shell that future lessons will
fill in.

## Step 6: Run A Shape Check

Use:

```python
config = GPTConfig()
model = TinyGPTShell(config)

x = torch.tensor([
    [1, 2, 3, 4],
    [5, 1, 2, 3],
], dtype=torch.long)

logits = model(x)

print(logits.shape)
```

Expected output:

```text
torch.Size([2, 4, 6])
```

This proves the model shell has the right input and output shapes:

```text
input ids:
    [B, T]

logits:
    [B, T, vocab_size]
```

That shape contract is what the loss function needs next.

## Step 7: Understand What This Model Cannot Do Yet

This shell can produce logits, but it cannot really use context.

Why?

Because there is no attention yet.

The vector at position 3 does not know what happened at positions 0, 1, or 2.
It only knows its own token id.

That is fine for this lesson.

We are not claiming this is a complete GPT. We are creating a clean frame:

```text
token ids in
logits out
```

Next, we can plug in:

```text
RMSNorm
RoPE attention
SwiGLU MLP
transformer blocks
```

without changing the outside contract.

## Step 8: Optional Weight Tying

Many language models tie the token embedding weights and LM head weights.

That means the same matrix used to read token ids into vectors is also used to
project vectors back to vocabulary scores.

The shapes line up:

```text
token_embedding.weight.shape = [vocab_size, d_model]
lm_head.weight.shape         = [vocab_size, d_model]
```

In PyTorch:

```python
self.lm_head.weight = self.token_embedding.weight
```

For this course, keep weight tying in mind but do not depend on it yet. The
important part right now is understanding the two roles:

```text
token embedding:
    token id -> vector

LM head:
    vector -> vocabulary logits
```

## Step 9: Write Your Lesson 04 Notes

Write:

```markdown
# Lesson 04 Notes

## Token Embedding

Input shape:
Embedding table shape:
Output shape:

## LM Head

Input shape:
Output shape:

## Forward Pass

Fill in:

input_ids:
h = token_embedding(input_ids):
logits = lm_head(h):

## What Is Missing

Why is this not a full GPT yet?
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain what `nn.Embedding` does
- you can trace `[B, T] -> [B, T, d_model]`
- you can explain what the LM head does
- you can trace `[B, T, d_model] -> [B, T, vocab_size]`
- you know this is only the GPT shell, not the full context-mixing model yet

Stop here. [Lesson 05](05_next_token_cross_entropy_loss.md) turns logits into next-token loss.
