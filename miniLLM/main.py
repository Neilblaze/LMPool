# pip install torch transformers datasets rotary-embedding-torch

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import _LRScheduler

import math
from transformers import AutoTokenizer
from dataclasses import dataclass
from datasets import load_dataset
from rotary_embedding_torch import RotaryEmbedding


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    mla: bool


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, rotary_emb_fn, cache=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.args.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.args.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.args.head_dim).transpose(1, 2)

        if cache is not None:
            key_cache, value_cache = cache
            # Keys are stored unrotated so the full sequence can be re-rotated each step.
            # Concatenate on seq dim (dim=2) since tensors are [b, n_heads, t, d].
            key_unrotated = torch.cat([key_cache, xk], dim=2)
            value = torch.cat([value_cache, xv], dim=2)
            new_cache = (key_unrotated, value)
            xq, key = rotary_emb_fn.rotate_queries_with_cached_keys(xq, key_unrotated)
        else:
            new_cache = (xk, xv)
            xq = rotary_emb_fn.rotate_queries_or_keys(xq)
            xk = rotary_emb_fn.rotate_queries_or_keys(xk)
            key, value = xk, xv

        key = torch.repeat_interleave(key, repeats=self.repeats, dim=1)
        value = torch.repeat_interleave(value, repeats=self.repeats, dim=1)

        output = F.scaled_dot_product_attention(
            xq, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
        )

        output = output.transpose(1, 2).view(batch_size, seq_len, self.n_heads * self.args.head_dim)
        return self.wo(output), new_cache


class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim

        self.q_rank = args.dim // 4
        self.kv_rank = args.dim // 4

        self.qk_nope_head_dim = self.head_dim
        self.qk_rope_head_dim = self.head_dim // 2
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.dq = nn.Linear(args.dim, self.q_rank, bias=False)
        self.q_norm = RMSNorm(self.q_rank, eps=args.norm_eps)
        self.uq = nn.Linear(self.q_rank, self.qk_head_dim * self.n_heads, bias=False)

        self.dkv_nope = nn.Linear(args.dim, self.kv_rank, bias=False)
        self.dkv_norm = RMSNorm(self.kv_rank, eps=args.norm_eps)
        self.uk_rope = nn.Linear(args.dim, self.qk_rope_head_dim, bias=False)
        self.uv = nn.Linear(self.kv_rank, self.n_heads * self.head_dim, bias=False)
        self.uk_nope = nn.Linear(self.kv_rank, self.n_heads * self.qk_nope_head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x, rotary_emb_fn, cache=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        kv_nope = self.dkv_norm(self.dkv_nope(x))
        k_rope = self.uk_rope(x)

        if cache is not None:
            cached_kv_nope, cached_k_rope = cache
            k_rope = torch.cat([cached_k_rope, k_rope], dim=1)
            kv_nope = torch.cat([cached_kv_nope, kv_nope], dim=1)

        new_cache = (kv_nope, k_rope)
        kv_seq_len = kv_nope.shape[1]

        k_nope = self.uk_nope(kv_nope)
        k_nope = k_nope.view(batch_size, kv_seq_len, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)

        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1).transpose(1, 2)

        v = self.uv(kv_nope)
        v = v.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.uq(self.q_norm(self.dq(x)))
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if cache is not None:
            q_rope, k_rope = rotary_emb_fn.rotate_queries_with_cached_keys(q_rope, k_rope)
        else:
            q_rope = rotary_emb_fn.rotate_queries_or_keys(q_rope)
            k_rope = rotary_emb_fn.rotate_queries_or_keys(k_rope)

        k_full = torch.cat([k_nope, k_rope], dim=-1)
        q_full = torch.cat([q_nope, q_rope], dim=-1)

        output = F.scaled_dot_product_attention(
            q_full, k_full, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
        )

        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.wo(output), new_cache


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MLA(args) if args.mla else Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, rotary_emb_fn, cache=None):
        h, new_cache = self.attention.forward(self.attention_norm(x), rotary_emb_fn, cache=cache)
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, new_cache


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args=args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.rotary_emb_fn = RotaryEmbedding(dim=args.head_dim // 2)

    def forward(self, input_ids: torch.Tensor, labels=None) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h, _ = layer(h, self.rotary_emb_fn, cache=None)
        logits = self.output(self.norm(h)).float()
        if labels is None:
            return logits
        loss = F.cross_entropy(logits.transpose(-1, -2), labels)
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200):
        cache = [None] * self.n_layers
        for _ in range(max_new_tokens):
            input_pos = idx[:, -1:] if idx.size(1) > 1 else idx
            h = self.tok_embeddings(input_pos)
            for i, layer in enumerate(self.layers):
                h, cache[i] = layer(h, self.rotary_emb_fn, cache=cache[i])
            logits = self.output(self.norm(h)).float()
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        if self.last_epoch > self.total_steps:
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]
        decay_ratio = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 - self.min_lr_ratio) * (1 + math.cos(math.pi * decay_ratio))
        return [base_lr * (self.min_lr_ratio + coeff) for base_lr in self.base_lrs]


class StreamingDataset(IterableDataset):
    def __init__(self, stream_dataset, tokenizer, seq_len):
        self.stream_dataset = stream_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        for sample in self.stream_dataset:
            tokenized = self.tokenizer(
                sample["text"],
                truncation=True,
                padding="max_length",
                max_length=self.seq_len,
                return_tensors="pt",
            )
            yield tokenized["input_ids"].squeeze(0)


def main():
    TOTAL_STEPS        = 1e5
    WARMUP_STEPS       = 300
    LEARNING_RATE      = 1e-4
    MAX_GRAD_CLIP_NORM = 1.0
    BATCH_SIZE         = 16
    SEQ_LEN            = 512 + 1
    EPOCHS             = 3
    PRINT_LOSS         = 10
    GENERATE_EVERY     = 100

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", pad_token="[PAD]")
    VOCAB_SIZE = len(tokenizer)

    dataset = load_dataset("c4", "en", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    train_dataset = StreamingDataset(dataset["train"],      tokenizer, SEQ_LEN)
    valid_dataset = StreamingDataset(dataset["validation"], tokenizer, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, drop_last=True)
    valid_loader_iterator = iter(valid_loader)

    args = ModelArgs(
        dim=768,
        n_layers=12,
        head_dim=64,
        hidden_dim=2048,  # (4 * dim) for regular FF, (8/3 * dim) for GLU variants
        n_heads=8,
        n_kv_heads=4,
        norm_eps=1e-5,
        vocab_size=VOCAB_SIZE,
        mla=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(args).to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = WarmupCosineLR(optimizer, warmup_steps=WARMUP_STEPS, total_steps=TOTAL_STEPS)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_loader):
            model.train()
            seq, labels = data[:, :-1], data[:, 1:]
            seq, labels = seq.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(seq, labels=labels)

            train_loss = loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if not (i % PRINT_LOSS):
                print(f"epoch {epoch}  step {i:>6d}  |  train loss: {train_loss:.4f}")

            if not (i % GENERATE_EVERY):
                model.eval()
                for _ in range(2):
                    context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    res = model.generate(context, max_new_tokens=100)
                    print(tokenizer.decode(res[0].tolist()), "\n")
                model.train()

    model.eval()
    for _ in range(3):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        res = model.generate(context, max_new_tokens=250)
        print(tokenizer.decode(res[0].tolist()), "\n")


if __name__ == "__main__":
    main()
