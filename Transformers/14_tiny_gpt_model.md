# Lesson 14: Full Tiny GPT Model

This lesson combines the pieces into one model:

```text
token embedding
transformer blocks
final RMSNorm
LM head
loss
generate
```

This is the first full modern GPT skeleton.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: Define The Model Fields

```python
class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

Fields:

```text
token_embedding:
    token ids -> vectors

blocks:
    repeated modern transformer blocks

final_norm:
    normalize before logits

lm_head:
    vectors -> vocabulary scores
```

## Step 2: Write The Forward Pass

```python
def forward(self, input_ids, targets=None):
    h = self.token_embedding(input_ids)

    for block in self.blocks:
        h = block(h)

    h = self.final_norm(h)
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

Shape flow:

```text
input_ids: [B, T]
h:         [B, T, d_model]
blocks:    [B, T, d_model]
logits:    [B, T, vocab_size]
```

## Step 3: Add Generation

```python
@torch.no_grad()
def generate(self, input_ids, max_new_tokens):
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -self.config.context_length:]
        logits, _ = self(input_cond)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids
```

This reuses the same forward pass.

Training and generation now share the same model.

## Step 4: Optional Weight Tying

You can tie input and output token weights:

```python
self.lm_head.weight = self.token_embedding.weight
```

This is common in language models.

It means:

```text
the same token table is used for reading tokens in and scoring tokens out
```

Keep it optional for now.

## Step 5: Run A Shape Check

```python
config = GPTConfig()
model = TinyGPT(config)

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

Expected:

```text
torch.Size([2, 4, 6])
tensor(...)
```

Now the model uses context through attention.

## Step 6: Know What You Have Built

The model now has the modern decoder-only structure:

```text
RMSNorm
RoPE causal self-attention
SwiGLU MLP
residual connections
stacked transformer blocks
final norm
LM head
```

The numbers are tiny, but the architecture path is real.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can trace the full model forward pass
- you know where transformer blocks sit
- you know where final RMSNorm sits
- you can compute logits and loss
- you can generate tokens with the same model

Stop here. [Lesson 15](15_training_loop.md) trains the model.
