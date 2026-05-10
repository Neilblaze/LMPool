# Lesson 02: Training Batches For Next-Token Prediction

[Lesson 01](01_modern_gpt_overview.md) gave the architecture map.

This lesson builds the data that feeds the model.

A decoder-only GPT trains on examples shaped like this:

```text
x: current tokens
y: next tokens
```

In tensor form:

```text
x.shape = [B, T]
y.shape = [B, T]
```

Where:

```text
B = batch size
T = context length
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Start With Token IDs

Text is converted into token ids before the model sees it.

Use a tiny vocabulary:

```text
0 = <pad>
1 = the
2 = cat
3 = sat
4 = on
5 = mat
```

The text:

```text
the cat sat on the mat
```

becomes:

```text
[1, 2, 3, 4, 1, 5]
```

Real tokenizers are more complex, but the model still receives integer ids.

## Step 2: Create x And y By Shifting

Next-token prediction uses a one-token shift.

From:

```text
tokens = [1, 2, 3, 4, 1, 5]
```

create:

```text
x = [1, 2, 3, 4, 1]
y = [2, 3, 4, 1, 5]
```

Read it position by position:

```text
1 predicts 2
2 predicts 3
3 predicts 4
4 predicts 1
1 predicts 5
```

That is the whole training target idea.

The target is not manually labeled. It is the next token in the stream.

## Step 3: Use A Context Length

The context length says how many tokens the model sees at once.

Let:

```text
T = 4
```

To create one example of length `T`, you need `T + 1` raw tokens.

Example raw window:

```text
[1, 2, 3, 4, 1]
```

Split it into:

```text
x = [1, 2, 3, 4]
y = [2, 3, 4, 1]
```

Why `T + 1`?

Because every input position needs a next-token target.

## Step 4: Batch Multiple Windows

Training one window at a time is inefficient, so we train on a batch.

Let:

```text
B = 2
T = 4
```

Two examples might be:

```text
x = [
  [1, 2, 3, 4],
  [5, 1, 2, 3],
]

y = [
  [2, 3, 4, 1],
  [1, 2, 3, 4],
]
```

The shapes are:

```text
x.shape = [2, 4]
y.shape = [2, 4]
```

Each row is one training example.

Each column is one time position.

## Step 5: Implement get_batch

Here is the first useful function:

```python
import torch


def get_batch(data: torch.Tensor, batch_size: int, context_length: int):
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack([
        data[i : i + context_length]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1 : i + context_length + 1]
        for i in starts
    ])

    return x, y
```

The input row uses:

```python
data[i : i + context_length]
```

The target row starts one token later:

```python
data[i + 1 : i + context_length + 1]
```

That one-token difference is the entire reason this function exists.

## Step 6: Check One Batch By Hand

Run:

```python
data = torch.tensor([1, 2, 3, 4, 1, 5, 1, 2, 3], dtype=torch.long)

x, y = get_batch(data, batch_size=2, context_length=4)

print(x)
print(y)
print(x.shape, y.shape)
```

The rows may change because starts are random.

The shapes should be:

```text
torch.Size([2, 4]) torch.Size([2, 4])
```

Now pick one row.

Example:

```text
x row = [4, 1, 5, 1]
y row = [1, 5, 1, 2]
```

Check it:

```text
4 predicts 1
1 predicts 5
5 predicts 1
1 predicts 2
```

If every `y` token is the next token after the matching `x` token, the batch is
correct.

## Step 7: Know What The Model Will Do With x And y

The model receives:

```text
x.shape = [B, T]
```

It produces:

```text
logits.shape = [B, T, vocab_size]
```

The loss compares:

```text
logits[b, t]
```

against:

```text
y[b, t]
```

So every position becomes a training question:

```text
given tokens up to here, what comes next?
```

## Step 8: Write Your Lesson 02 Notes

Write:

```markdown
# Lesson 02 Notes

## get_batch

What does it return?

## Shapes

B =
T =
x.shape =
y.shape =

## Shift

Why does y start one token after x?

## Manual Row Check

x row:
y row:
Explain the pairs:
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain why one raw window needs `T + 1` tokens
- you can explain why `x` and `y` both have shape `[B, T]`
- you can verify one row by hand
- you understand that `y[b, t]` is the correct next token for `x[b, t]`

Stop here. [Lesson 03](03_model_config_and_shapes.md) defines the model config and shape flow.
