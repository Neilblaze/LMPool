# Lesson 06: Generate Tokens One At A Time

[Lesson 05](05_next_token_cross_entropy_loss.md) gave the model a training interface:

```text
model(x, y) -> logits, loss
```

For generation, targets are not available.

Generation uses:

```text
model(x) -> logits
```

Then it chooses one next token, appends it, and repeats.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Know The Generation Loop

Generation is an autoregressive loop:

```text
start with token ids
run the model
take logits at the last position
turn logits into probabilities
sample one token
append the token
repeat
```

If the prompt is:

```text
[1, 2]
```

and the model samples token `3`, the sequence becomes:

```text
[1, 2, 3]
```

Then the model uses `[1, 2, 3]` to predict the next token.

## Step 2: Use Only The Last Position

The model returns logits for every position:

```text
logits.shape = [B, T, vocab_size]
```

During generation, only the final position predicts the next new token:

```python
next_logits = logits[:, -1, :]
```

Shape:

```text
[B, T, vocab_size] -> [B, vocab_size]
```

The earlier positions already predicted earlier tokens. The new token comes from
the last position.

## Step 3: Convert Logits To Probabilities

Use softmax:

```python
probs = torch.softmax(next_logits, dim=-1)
```

If:

```text
next_logits.shape = [B, vocab_size]
```

then:

```text
probs.shape = [B, vocab_size]
```

Each row sums to `1`.

Softmax turns arbitrary scores into a probability distribution.

## Step 4: Sample The Next Token

Use:

```python
next_id = torch.multinomial(probs, num_samples=1)
```

Shape:

```text
next_id.shape = [B, 1]
```

This samples one token id per batch row.

Sampling is different from always taking the highest score. Sampling allows
variety.

Later you can add temperature or top-k sampling, but keep the first version
simple.

## Step 5: Append The Token

Append along the time dimension:

```python
input_ids = torch.cat([input_ids, next_id], dim=1)
```

Shape:

```text
[B, T] + [B, 1] -> [B, T + 1]
```

Now the new token becomes part of the context for the next step.

## Step 6: Crop To Context Length

The model has a maximum context length:

```text
config.context_length
```

If the generated sequence becomes longer, crop to the most recent tokens:

```python
input_cond = input_ids[:, -config.context_length:]
```

This keeps:

```text
the latest context_length tokens
```

The newest tokens matter most for the next prediction, and the model was only
trained to handle sequences up to its context length.

## Step 7: Write The Generate Function

Add this method to the model:

```python
@torch.no_grad()
def generate(self, input_ids: torch.Tensor, max_new_tokens: int):
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -self.config.context_length:]
        logits, _ = self(input_cond)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return input_ids
```

Important lines:

```python
input_cond = input_ids[:, -self.config.context_length:]
```

This crops long prompts.

```python
next_logits = logits[:, -1, :]
```

This selects the last-position prediction.

```python
input_ids = torch.cat([input_ids, next_id], dim=1)
```

This appends the sampled token.

## Step 8: Run A Tiny Check

Use:

```python
prompt = torch.tensor([[1, 2]], dtype=torch.long)
out = model.generate(prompt, max_new_tokens=5)
print(out)
print(out.shape)
```

Expected shape:

```text
torch.Size([1, 7])
```

Why `7`?

```text
2 prompt tokens + 5 generated tokens
```

The output will not be meaningful yet. The model is still random and has no
attention.

This lesson is only about the generation mechanics.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain why generation uses only `logits[:, -1, :]`
- you can explain why the context is cropped
- you can explain softmax and sampling at a high level
- you can trace `[B, T] -> [B, T + 1]` after appending one token

Stop here. [Lesson 07](07_rmsnorm.md) builds RMSNorm.
