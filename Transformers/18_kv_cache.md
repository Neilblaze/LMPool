# Lesson 18: KV Cache

Generation is slow if the model recomputes the whole prefix every time.

KV cache makes generation faster by reusing previous keys and values.

This lesson teaches the concept.

You do not need to optimize training with KV cache.

KV cache is mainly for inference.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Step 1: See The Problem

Without cache:

```text
prompt: [the]
run model on [the]

append cat
run model on [the, cat]

append sat
run model on [the, cat, sat]
```

The old tokens are recomputed again and again.

That is wasteful.

## Step 2: Remember What Attention Uses

Attention creates:

```text
q, k, v
```

For generation, the new token needs to attend to previous tokens.

Previous keys and values do not change.

So we can store them.

```text
cache keys
cache values
```

Then each new step only computes q, k, v for the newest token.

## Step 3: Cache Shape

For one layer:

```text
k_cache.shape = [B, n_heads, past_T, head_dim]
v_cache.shape = [B, n_heads, past_T, head_dim]
```

For a new token:

```text
k_new.shape = [B, n_heads, 1, head_dim]
v_new.shape = [B, n_heads, 1, head_dim]
```

Append:

```python
k_all = torch.cat([k_cache, k_new], dim=2)
v_all = torch.cat([v_cache, v_new], dim=2)
```

Now attention uses:

```text
q_new attends over k_all and v_all
```

## Step 4: Know What Changes With RoPE

RoPE depends on position.

During cached generation, the new token is not always position `0`.

If the cache length is:

```text
past_T = 12
```

then the new token position is:

```text
position = 12
```

So RoPE must use the correct absolute position for the new q and k.

This is the main gotcha.

KV cache and RoPE work together, but position indices must be correct.

## Step 5: Training vs Cached Inference

Training:

```text
process full [B, T] sequences
compute loss at all positions
do not use KV cache
```

Cached generation:

```text
process one new token at a time
reuse old k and v
only need logits for the latest position
```

This is why KV cache belongs later in the course.

You first need normal attention to work.

Then you can optimize inference.

## Step 6: Minimal Cache Interface

A simple attention forward can accept:

```python
def forward(self, x, kv_cache=None, start_pos=0):
```

Return:

```python
return out, new_cache
```

Where:

```text
new_cache = {
    "k": k_all,
    "v": v_all,
}
```

Every transformer layer needs its own cache.

For a full model:

```text
cache = list of layer caches
```

## Step 7: Know The Payoff

Without KV cache, generating `N` tokens repeatedly recomputes previous tokens.

With KV cache, each step mostly processes the newest token and attends over
stored keys and values.

This is one reason real LLM serving systems can generate interactively.

The model architecture is the same.

The inference path is smarter.

![breaker](https://user-images.githubusercontent.com/48355572/209539106-8e1cbfc6-2f3d-4afd-b96a-890d967dd9ab.png)

## Done Checklist

You are done when:

- you can explain why generation recomputes old tokens without cache
- you know keys and values can be reused
- you can describe cache shapes
- you know RoPE needs correct positions during cached generation
- you know KV cache is for inference, not normal training

Stop here. You now have the full course path from batches to a modern GPT and its inference optimization.
