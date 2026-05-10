# Lesson 05: Next-Token Cross Entropy Loss

[Lesson 04](04_token_embeddings_and_lm_head.md) built a model shell:

```text
input ids [B, T] -> logits [B, T, vocab_size]
```

Now we need to train it.

Training requires one number:

```text
loss
```

The loss tells us how wrong the model was at predicting the next token.

For decoder-only GPT training, the standard loss is cross entropy over the next
token targets.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Start With Logits And Targets

The model returns logits:

```text
logits.shape = [B, T, vocab_size]
```

The batch gives targets:

```text
targets.shape = [B, T]
```

Example:

```text
B = 2
T = 4
vocab_size = 6
```

So:

```text
logits.shape  = [2, 4, 6]
targets.shape = [2, 4]
```

Read one position:

```text
logits[0, 2]  = 6 scores
targets[0, 2] = the correct next-token id
```

The target is one integer.

The logits are one score per possible token.

## Step 2: Understand One Position

Suppose:

```text
target token id = 3
```

The model produces:

```text
logits = [
  score for token 0,
  score for token 1,
  score for token 2,
  score for token 3,
  score for token 4,
  score for token 5,
]
```

Cross entropy asks:

```text
how much probability did the model assign to the correct token id?
```

If token `3` gets a high score, loss is lower.

If token `3` gets a low score, loss is higher.

You do not manually apply softmax before `F.cross_entropy`. PyTorch does that
internally in a numerically stable way.

## Step 3: Know The PyTorch Shape Requirement

PyTorch cross entropy expects:

```text
logits:  [N, vocab_size]
targets: [N]
```

But our model has:

```text
logits:  [B, T, vocab_size]
targets: [B, T]
```

So we flatten the batch and time dimensions together:

```text
B * T = N
```

For:

```text
B = 2
T = 4
```

we get:

```text
N = 8
```

The reshape is:

```text
logits:  [2, 4, 6] -> [8, 6]
targets: [2, 4]    -> [8]
```

This does not change the meaning. It just turns every batch/time position into
one row for the loss function.

## Step 4: Flatten Logits And Targets

In code:

```python
import torch.nn.functional as F


B, T, vocab_size = logits.shape

loss = F.cross_entropy(
    logits.reshape(B * T, vocab_size),
    targets.reshape(B * T),
)
```

The first argument:

```python
logits.reshape(B * T, vocab_size)
```

means:

```text
make one row per prediction position
keep one column per vocabulary token
```

The second argument:

```python
targets.reshape(B * T)
```

means:

```text
make one correct token id per prediction position
```

That is exactly what next-token training needs.

## Step 5: Connect Positions To Flattened Rows

Start with:

```text
targets = [
  [2, 3, 4, 1],
  [1, 2, 3, 4],
]
```

Flattened:

```text
targets_flat = [2, 3, 4, 1, 1, 2, 3, 4]
```

The matching logits flatten the same way:

```text
logits[0, 0] -> row 0
logits[0, 1] -> row 1
logits[0, 2] -> row 2
logits[0, 3] -> row 3
logits[1, 0] -> row 4
logits[1, 1] -> row 5
logits[1, 2] -> row 6
logits[1, 3] -> row 7
```

So:

```text
row 0 tries to predict token 2
row 1 tries to predict token 3
row 2 tries to predict token 4
...
```

Flattening is not losing sequence information. The model already produced one
prediction per position. The loss simply scores all positions as a list of
prediction tasks.

## Step 6: Add Loss To The Model Forward

Update the shell from [Lesson 04](04_token_embeddings_and_lm_head.md):

```python
import torch
import torch.nn.functional as F
from torch import nn


class TinyGPTShell(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        h = self.token_embedding(input_ids)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            B, T, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, vocab_size),
                targets.reshape(B * T),
            )

        return logits, loss
```

The model can now be used in two modes.

Training mode:

```python
logits, loss = model(x, y)
```

Generation mode later:

```python
logits, loss = model(x)
```

When targets are absent, loss stays `None`.

## Step 7: Run A Tiny Loss Check

Use:

```python
config = GPTConfig()
model = TinyGPTShell(config)

x = torch.tensor([
    [1, 2, 3, 4],
    [5, 1, 2, 3],
], dtype=torch.long)

y = torch.tensor([
    [2, 3, 4, 1],
    [1, 2, 3, 4],
], dtype=torch.long)

logits, loss = model(x, y)

print(logits.shape)
print(loss)
```

Expected shape:

```text
torch.Size([2, 4, 6])
```

The loss should be one scalar tensor:

```text
tensor(...)
```

The exact number is not important yet because the model weights are random.

The important part is:

```text
the model can produce logits
the loss can compare logits to next-token targets
```

## Step 8: Understand A Reasonable Starting Loss

If the model is guessing randomly across `vocab_size` tokens, the loss starts
near:

```text
log(vocab_size)
```

For:

```text
vocab_size = 6
```

that is about:

```text
log(6) = 1.79
```

Do not expect the exact number every time. Random initialization can make it
higher or lower.

This gives you a rough sanity check:

```text
tiny vocab random model:
    loss near 1.8 is unsurprising

large vocab random model:
    loss near log(vocab_size) is unsurprising
```

Later, when training works, the loss should decrease.

## Step 9: Know What Loss Does Not Prove Yet

A loss value proves that the tensors line up.

It does not prove the model is smart.

Right now, the shell still cannot mix context because we have not built attention.

That is okay.

The purpose of this lesson is to make the training interface work:

```text
x, y -> logits, loss
```

Every future architecture piece will keep this same interface.

## Step 10: Write Your Lesson 05 Notes

Write:

```markdown
# Lesson 05 Notes

## Shapes Before Loss

logits:
targets:

## Shapes For Cross Entropy

logits after flatten:
targets after flatten:

## One Prediction Row

Pick one [b, t] position.

target token id:
which logits row after flattening:

## Training Interface

model(x, y) returns:
model(x) returns:
```

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain why logits are `[B, T, vocab_size]`
- you can explain why targets are `[B, T]`
- you can flatten logits to `[B*T, vocab_size]`
- you can flatten targets to `[B*T]`
- you understand that cross entropy scores every next-token prediction position

Stop here. [Lesson 06](06_generate_tokens_one_at_a_time.md) will use logits to generate tokens one at a time.
