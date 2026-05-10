# Lesson 01: Modern Decoder-Only GPT Overview

This course builds a small modern GPT-style language model from scratch.

Not an encoder.
Not an old tutorial transformer with learned absolute position embeddings.
Not a bigram detour.

We are building the core shape used by modern decoder-only LLMs:

```text
token ids
-> token embeddings
-> repeated transformer blocks
-> final norm
-> vocabulary logits
-> next-token loss or generation
```

Each transformer block will use:

```text
RMSNorm
causal self-attention with RoPE
residual connections
SwiGLU MLP
```

This first lesson is the map. You should finish it knowing what every major part
is responsible for before we write code.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Know The Job

A decoder-only GPT predicts the next token.

Example:

```text
input:
the cat sat on the

next token:
mat
```

During training, the model learns this at every position:

```text
given "the"              predict "cat"
given "the cat"          predict "sat"
given "the cat sat"      predict "on"
given "the cat sat on"   predict "the"
```

The labels are not separate classes written by a human. The labels come from the
same token stream shifted by one position.

That is the central training idea:

```text
x = current tokens
y = next tokens
```

## Step 2: See The Whole Model

A modern decoder-only GPT has this high-level structure:

```text
input token ids [B, T]

token embedding
    [B, T] -> [B, T, d_model]

transformer block 1
transformer block 2
...
transformer block N
    [B, T, d_model] stays [B, T, d_model]

final RMSNorm
    [B, T, d_model]

LM head
    [B, T, d_model] -> [B, T, vocab_size]
```

Read the shape symbols like this:

```text
B = batch size
T = context length
d_model = hidden size
vocab_size = number of possible token ids
```

The model starts with integer token ids and ends with one score per vocabulary
token at every position.

Those scores are called logits.

## Step 3: Understand The Residual Stream

The main tensor inside the model is often called the residual stream.

It has shape:

```text
x.shape = [B, T, d_model]
```

At first, `x` is just token embeddings.

After each transformer block, `x` becomes more contextual.

Position 0 can only represent what is known from position 0.

Position 5 can represent information from positions:

```text
0, 1, 2, 3, 4, 5
```

It cannot use future positions because decoder-only attention is causal.

## Step 4: Understand One Transformer Block

One modern GPT block looks like this:

```text
x = x + attention(rmsnorm(x))
x = x + swiglu_mlp(rmsnorm(x))
```

This is called pre-norm because normalization happens before attention and MLP.

The block has two jobs:

```text
attention:
    lets each token position read earlier token positions

MLP:
    transforms each token position after context has been mixed
```

The residual additions keep information flowing forward:

```text
x = x + something_new
```

The model does not replace the stream. It updates it.

## Step 5: Understand Causal Attention

Self-attention lets token positions communicate.

In a decoder-only model, the communication is one-way:

```text
a token can look left
a token can look at itself
a token cannot look right
```

For:

```text
[the, cat, sat]
```

the allowed pattern is:

```text
"the" can see:
    the

"cat" can see:
    the, cat

"sat" can see:
    the, cat, sat
```

This prevents cheating during training.

If `"the"` could see `"cat"` while trying to predict `"cat"`, the task would be
fake. The causal mask blocks that.

## Step 6: Understand RoPE

The model needs to understand token order.

Older GPT models often added learned position embeddings to token embeddings:

```text
x = token_embedding + position_embedding
```

We will not make that the main architecture.

We will use RoPE:

```text
RoPE = rotary position embeddings
```

RoPE applies position information inside attention. More specifically, it rotates
the query and key vectors based on position before attention scores are computed.

Keep the mental model simple for now:

```text
token embeddings say what token is present
RoPE helps attention know where tokens are relative to each other
```

RoPE does not change the high-level shape:

```text
[B, T, d_model] stays [B, T, d_model]
```

It changes how attention compares positions.

## Step 7: Understand RMSNorm

RMSNorm is a lightweight normalization layer used in many modern LLMs.

It keeps the scale of activations stable before attention and before the MLP:

```text
x = x + attention(rmsnorm(x))
x = x + mlp(rmsnorm(x))
```

You do not need the formula yet. The first mental model is:

```text
RMSNorm makes the vector scale easier for the next layer to work with
```

We will implement it directly later.

## Step 8: Understand SwiGLU

The MLP in many modern LLMs is not a plain ReLU MLP.

We will use SwiGLU.

The rough idea:

```text
one projection creates values
one projection creates a gate
the gate controls which values pass through
```

This gives the model a stronger per-token transformation after attention has
mixed context.

For now, remember the block:

```text
attention mixes positions
SwiGLU transforms each position
```

## Step 9: Understand Training vs Generation

Training uses known text and shifted targets:

```text
x: [the, cat, sat]
y: [cat, sat, on]
```

The model predicts all next tokens in parallel.

Generation uses the model repeatedly:

```text
start: "the"
predict: "cat"
append: "the cat"
predict: "sat"
append: "the cat sat"
```

Training teaches the model.

Generation uses the trained model to produce new tokens.

## Step 10: Write Your Lesson 01 Notes

Write this short note:

```markdown
# Lesson 01 Notes

## What We Are Building

One sentence:

## Transformer Block

Fill in:

x = x + attention(_____)
x = x + swiglu_mlp(_____)

## Key Parts

RMSNorm:
RoPE:
Causal attention:
SwiGLU:

## Training vs Generation

Training:
Generation:
```

Keep it short. The goal is to know the architecture we are building before we
start coding pieces.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain that GPT predicts next tokens
- you know the model uses RMSNorm, RoPE attention, residuals, and SwiGLU
- you can explain why causal attention blocks future tokens
- you know RoPE belongs inside attention, not as an added token embedding
- you can describe the difference between training and generation

Stop here. [Lesson 02](02_training_batches.md) builds the training batches that feed this model.
