# Speculative decoding with rejection sampling for autoregressive SLMs
#
# Reference: arXiv:2302.01318
# "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
#
# A small draft model proposes gamma tokens per step, the target model
# verifies them in a single forward pass, and a rejection sampling scheme
# guarantees the output distribution matches the target exactly.


import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from typing import Tuple, Dict, Any, List


def truncate_kv_cache(cache: DynamicCache, seq_len: int) -> None:
    """Truncate all layers of a DynamicCache to the first seq_len positions"""
    assert isinstance(cache, DynamicCache)
    for layer in cache.layers:
        layer.keys = layer.keys[..., :seq_len, :]
        layer.values = layer.values[..., :seq_len, :]


@dataclass
class SpeculativeConfig:
    gamma: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 128
    early_stop: bool = True


class SpeculativeDecoder:

    def __init__(self, target_model, draft_model, tokenizer, config: SpeculativeConfig, device: str = "cuda"):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.target_kv_cache = None
        self.draft_kv_cache = None
        self.next_token = None

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling, top-k, and nucleus (top-p) filtering"""
        if self.config.temperature > 0.0:
            scaled = logits / self.config.temperature

            if self.config.top_k > 0:
                topk_vals, topk_idx = torch.topk(scaled, self.config.top_k, dim=-1)
                filtered = torch.full_like(scaled, float("-inf"))
                filtered.scatter_(-1, topk_idx, topk_vals)
                scaled = filtered

            if self.config.top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                # mask everything past the nucleus threshold
                mask = cumsum > self.config.top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                remove = mask.scatter(-1, sorted_idx, mask)
                scaled = scaled.masked_fill(remove, float("-inf"))

            return F.softmax(scaled, dim=-1)

        return F.softmax(logits, dim=-1)

    def _sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a single token from logits [1, vocab]. Returns (token, full prob distribution)."""
        probs = self._logits_to_probs(logits)
        if self.config.temperature > 0.0:
            token = torch.multinomial(probs, num_samples=1)
        else:
            token = torch.argmax(probs, dim=-1, keepdim=True)
        return token, probs

    def _generate_draft_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate gamma tokens from the draft model with KV caching.

        Returns:
            draft_tokens:      [1, gamma]
            draft_token_probs: [1, gamma, vocab_size]
        """
        tokens: List[int] = []
        token_probs: List[torch.Tensor] = []
        current = self.next_token

        with torch.inference_mode():
            for _ in range(self.config.gamma):
                outputs = self.draft_model(
                    current,
                    use_cache=True,
                    past_key_values=self.draft_kv_cache,
                    return_dict=False,
                )
                logits = outputs[0][:, -1, :]
                current, probs = self._sample(logits)
                tokens.append(current.item())
                token_probs.append(probs)

        draft_tokens = torch.tensor([tokens], device=self.device, dtype=torch.long)
        draft_token_probs = torch.stack(token_probs, dim=0).transpose(0, 1)
        return draft_tokens, draft_token_probs

    def _verify_draft_tokens(
        self, draft_tokens: torch.Tensor, draft_token_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Run all draft tokens + next_token through the target in one forward pass,
        then accept/reject each draft token using the speculative sampling criterion."""
        accepted_ids: List[int] = []
        num_accepted = 0

        with torch.inference_mode():
            # target sees [next_token, draft_0, ..., draft_{gamma-1}] in a single pass
            verify_seq = torch.cat([self.next_token, draft_tokens], dim=1)
            outputs = self.target_model(
                verify_seq,
                use_cache=True,
                past_key_values=self.target_kv_cache,
                return_dict=False,
            )
            target_logits = outputs[0]                                  # [1, gamma+1, vocab]
            target_probs = self._logits_to_probs(target_logits)         # [1, gamma+1, vocab]

            for i in range(self.config.gamma):
                draft_tok = draft_tokens[:, i : i + 1]                  # [1, 1]
                p_draft = draft_token_probs[:, i, :].gather(-1, draft_tok).squeeze()
                p_target = target_probs[:, i, :].gather(-1, draft_tok).squeeze()

                if self.config.temperature == 0.0:
                    # deterministic: accept iff target argmax agrees
                    target_tok = torch.argmax(target_probs[:, i, :], dim=-1)
                    if target_tok.item() == draft_tok.item():
                        accepted_ids.append(draft_tok.item())
                        num_accepted += 1
                        continue
                else:
                    # stochastic: accept with probability min(p_target / p_draft, 1)
                    accept_ratio = min((p_target / p_draft).item(), 1.0)
                    if torch.rand(1, device=self.device).item() < accept_ratio:
                        accepted_ids.append(draft_tok.item())
                        num_accepted += 1
                        continue
                break

            # sample the corrective / bonus token
            if num_accepted < self.config.gamma:
                # rejection: sample from clamp(p_target - p_draft, 0), re-normalized
                adjusted = torch.clamp(
                    target_probs[:, num_accepted, :] - draft_token_probs[:, num_accepted, :],
                    min=0,
                )
                next_probs = adjusted / adjusted.sum(dim=-1, keepdim=True)
            else:
                # all accepted: bonus token from target's distribution at position gamma
                next_probs = target_probs[:, num_accepted, :]

            if self.config.temperature == 0.0:
                next_token = torch.argmax(next_probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(next_probs, num_samples=1)

        return (
            torch.tensor([accepted_ids], device=self.device, dtype=torch.long),
            num_accepted,
            next_token,
        )

    def decode(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Run speculative decoding on a prompt. Returns (completion, metrics)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        generated_tokens = []

        # prefill: run the full prompt through both models to seed their KV caches
        with torch.inference_mode():
            draft_output = self.draft_model(
                input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=False,
            )
            self.draft_kv_cache = draft_output[1]

        with torch.inference_mode():
            target_output = self.target_model(
                input_ids, attention_mask=attention_mask,
                use_cache=True, return_dict=False,
            )
            self.target_kv_cache = target_output[1]

        seq_len = input_ids.size(1)

        # first token from target
        first_logits = target_output[0][:, -1, :]
        self.next_token, _ = self._sample(first_logits)
        generated_tokens.append(self.next_token.item())

        total_proposed = 0
        total_accepted = 0

        while seq_len + 1 < self.config.max_new_tokens:
            draft_tokens, draft_token_probs = self._generate_draft_tokens()
            accepted_tokens, num_accepted, next_token = self._verify_draft_tokens(
                draft_tokens, draft_token_probs
            )

            total_proposed += self.config.gamma
            total_accepted += num_accepted

            # advance the KV caches: keep prompt + all generated tokens so far
            seq_len += 1 + num_accepted
            if num_accepted == self.config.gamma:
                # all draft tokens accepted; push the last accepted token through
                # the draft model so its cache stays in sync
                with torch.inference_mode():
                    self.draft_model(
                        accepted_tokens[:, -1:],
                        use_cache=True,
                        past_key_values=self.draft_kv_cache,
                        return_dict=False,
                    )
            else:
                truncate_kv_cache(self.draft_kv_cache, seq_len)
            truncate_kv_cache(self.target_kv_cache, seq_len)

            generated_tokens.extend(accepted_tokens[0].tolist())
            self.next_token = next_token
            generated_tokens.append(self.next_token.item())

            if self.config.early_stop and self.tokenizer.eos_token_id in generated_tokens:
                eos_idx = generated_tokens.index(self.tokenizer.eos_token_id)
                generated_tokens = generated_tokens[:eos_idx]
                break

        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        metrics = {
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "acceptance_ratio": total_accepted / total_proposed if total_proposed > 0 else 0,
            "generated_length": len(generated_tokens),
        }
        return completion, metrics


def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 128,
                  device: str = "cuda") -> Tuple[str, int]:
    """Baseline: greedy decoding via model.generate()"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.size(1)

    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def greedy_decode_with_cache(model, tokenizer, prompt: str, max_new_tokens: int = 128,
                             device: str = "cuda") -> Tuple[str, int]:
    """Baseline: manual greedy loop with explicit KV caching."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_tokens = []

    with torch.inference_mode():
        # prefill
        output = model(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            use_cache=True, return_dict=False,
        )
        kv_cache = output[1]
        next_token = torch.argmax(output[0][:, -1, :], dim=-1)
        generated_tokens.append(next_token.item())

        # decode
        while len(generated_tokens) < max_new_tokens:
            output = model(
                next_token.view(1, 1),
                use_cache=True,
                past_key_values=kv_cache,
                return_dict=False,
            )
            next_token = torch.argmax(output[0][:, -1, :], dim=-1)
            generated_tokens.append(next_token.item())

            if tokenizer.eos_token_id in generated_tokens:
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True), len(generated_tokens)


if __name__ == "__main__":
    target_name = "Qwen/Qwen3-4B"  # recommend 100x larger than draft model
    draft_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading target={target_name}, draft={draft_name} on {device}")
    target_model = AutoModelForCausalLM.from_pretrained(target_name).to(device).eval()
    draft_model = AutoModelForCausalLM.from_pretrained(draft_name).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(target_name)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name)
    assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), "Vocabularies must match between target and draft"

    prompt = "introduce yourself"
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    config = SpeculativeConfig(
        gamma=3,               # number of draft tokens to generate each iteration
        temperature=0.0,       # greedy for deterministic comparison
        top_p=1,
        top_k=0,
        max_new_tokens=128,
        early_stop=True,
    )

    max_tokens = config.max_new_tokens


    # --- greedy baseline (model.generate) ---
    t0 = time.time()
    greedy_text, greedy_n = greedy_decode(target_model, tokenizer, formatted_prompt, max_tokens, device)
    greedy_time = time.time() - t0


    # --- greedy baseline (manual KV cache loop) ---
    t0 = time.time()
    cache_text, cache_n = greedy_decode_with_cache(target_model, tokenizer, formatted_prompt, max_tokens, device)
    cache_time = time.time() - t0


    # --- speculative decoding ---
    decoder = SpeculativeDecoder(target_model, draft_model, tokenizer, config, device=device)
    t0 = time.time()
    spec_text, metrics = decoder.decode(formatted_prompt)
    spec_time = time.time() - t0

    print(f"\nGreedy          : {greedy_n} tokens, {greedy_time:.2f}s, {greedy_n / greedy_time:.1f} tok/s")
    print(f"Greedy (cached) : {cache_n} tokens, {cache_time:.2f}s, {cache_n / cache_time:.1f} tok/s")
    print(f"Speculative     : {metrics['generated_length']} tokens, {spec_time:.2f}s, "
          f"{metrics['generated_length'] / spec_time:.1f} tok/s")
    print(f"\nAcceptance: {metrics['total_accepted']}/{metrics['total_proposed']} "
          f"({metrics['acceptance_ratio']:.1%})")
    print(f"Speedup vs greedy: {greedy_time / spec_time:.2f}x")
    print(f"Speedup vs cached: {cache_time / spec_time:.2f}x")