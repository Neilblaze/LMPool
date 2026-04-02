# microGPT with Attention Residuals

Minimal, dependency-free implementations of [Attention Residuals](https://arxiv.org/abs/2603.15031) (MoonshotAI) applied to [karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). Both variants from the paper are implemented in the same pure-Python, scalar-autograd style as the original.

| File | Variant | Candidate set |
|------|---------|---------------|
| [`micGPT_FAR.py`](micGPT_FAR.py) | Full AttnRes | Every individual sublayer output. O(2*L) candidates. |
| [`micGPT_BAR.py`](micGPT_BAR.py) | Block AttnRes | Grouped block summaries + current partial accumulator. O(blocks) candidates. |
| [`microgpt.py`](microgpt.py) | Baseline | Standard additive residual (no AttnRes). |

> [!NOTE]
> The original microgpt uses `n_layer = 1`, which means attention residuals only have 3 candidates (embedding, attention output, MLP output). The learned mixing is near-trivial at this depth. These implementations exist as barebones reference code to see the mechanism without framework abstractions, not as performance benchmarks.


## Background

In a standard transformer, the residual stream is a running sum. Each sublayer adds its output to whatever came before:

```python
x = x + attn(rmsnorm(x))
x = x + mlp(rmsnorm(x))
```

Attention Residuals change how the **input to each sublayer** is constructed. Instead of feeding the running sum directly, a learned projection vector scores all previous outputs via softmax, and the sublayer receives a weighted combination of those outputs. The additive accumulation is preserved for the sublayer *output*, but the sublayer *input* is a learned selective mix rather than an undifferentiated sum.

Three properties from the paper that are worth noting:

- **Scoring uses normalized candidates** (RMSNorm) but the **weighted sum uses raw values**
- **Projection vectors are zero-initialized**, so at init the softmax is uniform and the mechanism recovers standard residual behavior
- **Block boundaries** (in BAR) commit the current partial accumulator as a block summary and start a fresh accumulator


## Full Attention Residuals (FAR)

Full AttnRes collects every sublayer output into a growing candidate list `layer_outs`. Before each sublayer, a learned projection scores all candidates and produces a weighted mix as the sublayer input.

The key change from the baseline `gpt()` forward pass:

```python
# Baseline: additive residual
x_residual = x
x = rmsnorm(x)
# ... sublayer computation ...
x = [a + b for a, b in zip(x, x_residual)]

# Full AttnRes: learned mix over all sublayer outputs
w = softmax([
    sum(p * k for p, k in zip(proj[0], rmsnorm(r)))
    for r in layer_outs
])
x = [sum(w[i] * layer_outs[i][j] for i in range(len(w))) for j in range(n_embd)]
x = rmsnorm(x)
# ... sublayer computation ...
layer_outs.append(sublayer_output)
```

The candidate list grows as `[embedding, attn_out_0, mlp_out_0, attn_out_1, mlp_out_1, ...]`, one entry per sublayer across all layers.

After the final layer, a separate `out_res_proj` performs one more learned mix over all candidates before the language model head.


## Block Attention Residuals (BAR)

Full AttnRes has O(2*L) candidates, which becomes expensive at depth. Block AttnRes groups layers into blocks. Instead of `layer_outs`, it maintains two things:

- `blocks` - completed block summaries (committed at block boundaries)
- `partial_block` - the current block's running sum of sublayer outputs

At each block boundary (controlled by `layers_per_block`), the current `partial_block` is appended to `blocks` and a fresh accumulator starts. The candidate set for the learned mixing is always `blocks + [partial_block]`.

With `layers_per_block = 1` (the default), the execution trace for `n_layer = 2` looks like:

```
layer 0: mix [x] -> commit x -> attn -> mlp -> partial = attn_out_0 + mlp_out_0
layer 1: mix [x, partial_0] -> commit partial_0 -> attn -> mlp -> partial = attn_out_1 + mlp_out_1
final:   mix [x, partial_0, partial_1]
```


## Reference pseudocode from the paper

The paper provides this PyTorch-style reference for Block AttnRes:

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])      # [N+1, B, T, D]
    K = norm(V)
    logits = torch.einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = torch.einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h

def forward(self, blocks, hidden_states):
    partial_block = hidden_states

    # AttnRes before attention
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

    # Block boundary: commit partial, start fresh
    if self.layer_number % (self.block_size // 2) == 0:
        blocks.append(partial_block)
        partial_block = None

    # Self-attention
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out

    # AttnRes before MLP
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

    # MLP
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out

    return blocks, partial_block
```


## Usage

No dependencies. Just Python 3.

```bash
python micGPT_FAR.py   # Full Attention Residuals
python micGPT_BAR.py   # Block Attention Residuals
python microgpt.py     # Baseline (no AttnRes)
```

All three auto-download a names dataset (~32K names), train for 1000 steps with Adam, and generate 20 sample names.


## Credits

- Original microgpt by [Andrej Karpathy](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- Attention Residuals paper and code by [MoonshotAI](https://github.com/MoonshotAI/Attention-Residuals)