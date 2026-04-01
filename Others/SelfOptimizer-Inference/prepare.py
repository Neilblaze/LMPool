"""
Evaluation Harness >>> //////////////////
NOTE: The agent must NOT modify this file.
"""

import json, math, re, statistics, subprocess, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


# Configuration
MODEL_ID = "mlx-community/gemma-3-270m-it-4bit"
MAX_TOKENS = 256
N_RUNS = 3
N_WARMUP = 2
PPL_CEILING = 50.0
SANITY_FLOOR = 0.6
INF = math.inf


# Pre-compiled patterns (avoid recompilation per call)
_RE_CONST = re.compile(r"^([A-Z][A-Z_0-9]+)\s*=\s*(.+?)(?:\s*#.*)?$", re.MULTILINE)
_RE_FUNC = re.compile(r"^def (\w+)\(", re.MULTILINE)
_RE_IMPORT = re.compile(r"^(?:from|import)\s+(\S+)", re.MULTILINE)
_RE_48 = re.compile(r"\b48\b")


# Sanity checks (tuples iterate faster than lists, immutable)
_KW_TRANSFORMER = ("attention", "self-attention", "token", "layer")
_KW_COMPILER = ("dead code", "constant folding", "inlining", "loop", "common subexpression", "cse", "licm", "dce")
_KW_POEM = ("silicon", "chip", "electric", "current", "circuit")
_KW_CODE = ("subsequence", "lcs", "longest")


PROMPTS = (
    "Explain the transformer architecture in neural networks, covering attention, layers, and token embeddings.",
    (
        "LLVM uses analysis and transformation passes to optimize IR. Key passes include "
        "dead code elimination (DCE), constant folding, loop-invariant code motion (LICM), "
        "common subexpression elimination (CSE), and inlining. The pass manager schedules "
        "these to maximize optimization while minimizing compile time. Profile-guided "
        "optimization (PGO) further refines hot-path decisions.\n\n"
        "Summarize the key optimization passes in one sentence."
    ),
    "Solve step by step: A train travels A to B at 60 km/h and returns at 40 km/h. What is the average speed for the round trip?",
    "Write a short poem about silicon chips and electricity.",
    "Write a Python function to find the longest common subsequence of two strings.",
)


# Sanity checks as named functions (each lowercases the text once, not per-keyword)
def _check_transformer(t):
    tl = t.lower()
    return sum(kw in tl for kw in _KW_TRANSFORMER) >= 2

def _check_compiler(t):
    tl = t.lower()
    return sum(kw in tl for kw in _KW_COMPILER) >= 2

def _check_math(t):
    return bool(_RE_48.search(t))

def _check_poem(t):
    return t.count("\n") >= 2 and any(kw in t.lower() for kw in _KW_POEM)

def _check_code(t):
    return "def " in t and any(kw in t.lower() for kw in _KW_CODE)

SANITY_CHECKS = (_check_transformer, _check_compiler, _check_math, _check_poem, _check_code)


TRADEOFF_RULES = {
    "TEMP": lambda o, n: "Argmax decoding: no sampling overhead but deterministic output." if n in ("0.0", "0") else None,
    "KV_BITS": lambda o, n: f"KV cache quantized to {n}-bit. May degrade long-sequence quality." if n not in ("None", "(removed)") else None,
    "MAX_KV_SIZE": lambda o, n: f"Rotating KV cache ({n} tokens). Context lost beyond this window." if n not in ("None", "(removed)") else None,
    "MAX_TOKENS": lambda o, n: f"Token budget reduced ({o} -> {n}). Throughput gain may be artificial." if o.isdigit() and n.isdigit() and int(n) < int(o) else None,
    "PREFILL_STEP_SIZE": lambda o, n: f"Prefill step size changed ({o} -> {n}). May affect prefill throughput and peak memory." if o != n else None,
}


# Perplexity cache: with deterministic sampling (temp=0.0), the same prompt produces
# identical output across runs. Caching avoids redundant model forward passes.
# For N_RUNS=3 and 5 prompts, this saves 10 out of 15 forward passes.
_ppl_cache = {}


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _median(values):
    return statistics.median(values) if values else 0.0


def diff_from_baseline():
    # Report parameter and structural changes in inference.py vs its first commit
    try:
        log = subprocess.run(["git", "log", "--reverse", "--format=%H", "--", "inference.py"],
                             capture_output=True, text=True)
        hashes = log.stdout.strip().split("\n")
        if not hashes or not hashes[0]:
            return

        base = subprocess.run(["git", "show", f"{hashes[0]}:inference.py"],
                              capture_output=True, text=True).stdout
        with open("inference.py") as f:
            current = f.read()
        if base == current:
            return

        old_cfg, new_cfg = dict(_RE_CONST.findall(base)), dict(_RE_CONST.findall(current))

        deltas, warnings = [], []
        for key in sorted(old_cfg.keys() | new_cfg.keys()):
            ov, nv = old_cfg.get(key, "(absent)"), new_cfg.get(key, "(removed)")
            if ov != nv:
                deltas.append(f"  {key}: {ov} -> {nv}")
                rule = TRADEOFF_RULES.get(key)
                if rule:
                    w = rule(ov.strip(), nv.strip())
                    if w:
                        warnings.append(f"  - {w}")

        old_fns = set(_RE_FUNC.findall(base))
        new_fns = set(_RE_FUNC.findall(current))
        deltas += [f"  + {fn}()" for fn in sorted(new_fns - old_fns)]
        deltas += [f"  - {fn}()" for fn in sorted(old_fns - new_fns)]

        old_imports = set(_RE_IMPORT.findall(base))
        new_imports = set(_RE_IMPORT.findall(current))
        deltas += [f"  + import {mod}" for mod in sorted(new_imports - old_imports)]
        deltas += [f"  - import {mod}" for mod in sorted(old_imports - new_imports)]

        old_loc = sum(1 for l in base.splitlines() if l.strip())
        new_loc = sum(1 for l in current.splitlines() if l.strip())
        if old_loc != new_loc:
            d = new_loc - old_loc
            deltas.append(f"  LOC: {old_loc} -> {new_loc} ({'+' if d > 0 else ''}{d})")

        if deltas:
            print(f"\n[diff] changes from baseline:")
            for line in deltas:
                print(line)
            if warnings:
                print("[diff] tradeoff warnings:")
                for w in warnings:
                    print(w)
    except Exception:
        pass


def perplexity(model, tokenizer, text):
    # Next-token cross-entropy perplexity // Returns INF for degenerate outputs.
    # Cache hit avoids a full model forward pass (saves ~10 forward passes with deterministic sampling).
    if text in _ppl_cache:
        return _ppl_cache[text]

    ids = tokenizer.encode(text)
    if len(ids) < 2:
        _ppl_cache[text] = INF
        return INF

    tok = mx.array([ids])
    logits = model(tok)
    loss = nn.losses.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        tok[:, 1:].reshape(-1),
        reduction="none",
    )
    result = float(mx.exp(mx.mean(loss)).item())
    _ppl_cache[text] = result
    return result


def bench_one(fn, model, tokenizer, prompt, idx=-1):
    # Run a single prompt through the inference function and collect metrics
    t0 = time.perf_counter()
    out = fn(model, tokenizer, prompt)
    wall = time.perf_counter() - t0

    text = out.get("text", "")
    ppl = perplexity(model, tokenizer, text) if text.strip() else INF

    acc = None
    if 0 <= idx < len(SANITY_CHECKS):
        try:
            acc = SANITY_CHECKS[idx](text)
        except Exception:
            acc = False

    # Estimate time-to-first-token from prefill metrics
    prompt_tps = out.get("prompt_tps", 0.0)
    prompt_tokens = out.get("prompt_tokens", 0)
    ttft_ms = (prompt_tokens / prompt_tps * 1000) if prompt_tps > 0 else 0.0

    return {
        "prompt": (prompt[:80] + "...") if len(prompt) > 80 else prompt,
        "generation_tps": out.get("generation_tps", 0.0),
        "prompt_tps": prompt_tps,
        "peak_memory_gb": out.get("peak_memory_gb", 0.0),
        "wall_time_s": wall,
        "generation_tokens": out.get("generation_tokens", 0),
        "prompt_tokens": prompt_tokens,
        "perplexity": ppl,
        "text_length": len(text),
        "accurate": acc,
        "ttft_ms": ttft_ms,
    }


def evaluate(fn) -> dict:
    # Full evaluation: warmup, N_RUNS x len(PROMPTS) benchmark passes, aggregate.
    _ppl_cache.clear()

    print(f"Loading {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    # Warmup pass 1: short prompt for basic Metal kernel compilation
    # Warmup pass 2: actual benchmark prompt to prime kernels for real sequence lengths
    for w in range(N_WARMUP):
        wp = PROMPTS[0] if w > 0 else "Warmup."
        fn(model, tokenizer, wp)

    runs = []
    for r in range(N_RUNS):
        print(f"\n--- run {r + 1}/{N_RUNS} ---")
        batch = []
        for i, p in enumerate(PROMPTS):
            m = bench_one(fn, model, tokenizer, p, idx=i)
            batch.append(m)
            acc = f"  {'PASS' if m['accurate'] else 'FAIL'}" if m["accurate"] is not None else ""
            print(f"  gen={m['generation_tps']:.1f}  pfx={m['prompt_tps']:.1f}  "
                  f"mem={m['peak_memory_gb']:.2f}GB  ppl={m['perplexity']:.1f}  "
                  f"ttft={m['ttft_ms']:.1f}ms{acc}")
        runs.append(batch)

    # Single-pass metric collection (avoids 6+ iterations over the same data)
    gen_tps, prompt_tps, ttft, peak_mem = [], [], [], []
    valid_ppl, acc_flags = [], []
    for batch in runs:
        for r in batch:
            gen_tps.append(r["generation_tps"])
            prompt_tps.append(r["prompt_tps"])
            ttft.append(r["ttft_ms"])
            peak_mem.append(r["peak_memory_gb"])
            ppl = r["perplexity"]
            if ppl != INF:
                valid_ppl.append(ppl)
            acc = r["accurate"]
            if acc is not None:
                acc_flags.append(acc)

    avg_ppl = _mean(valid_ppl) if valid_ppl else INF
    sanity = _mean([1.0 if a else 0.0 for a in acc_flags]) if acc_flags else None
    passed = avg_ppl < PPL_CEILING and (sanity is None or sanity >= SANITY_FLOOR)

    summary = {
        "avg_generation_tps": round(_mean(gen_tps), 2),
        "std_generation_tps": round(_std(gen_tps), 2),
        "median_generation_tps": round(_median(gen_tps), 2),
        "avg_prompt_tps": round(_mean(prompt_tps), 2),
        "std_prompt_tps": round(_std(prompt_tps), 2),
        "avg_peak_memory_gb": round(_mean(peak_mem), 3),
        "avg_perplexity": round(avg_ppl, 2),
        "avg_ttft_ms": round(_mean(ttft), 2),
        "sanity_check": round(sanity, 2) if sanity is not None else None,
        "quality_pass": passed,
        "num_runs": N_RUNS,
        "num_prompts": len(PROMPTS),
    }

    print(f"\n[result] avg_generation_tps={summary['avg_generation_tps']} (±{summary['std_generation_tps']})  "
          f"median_generation_tps={summary['median_generation_tps']}  "
          f"avg_prompt_tps={summary['avg_prompt_tps']} (±{summary['std_prompt_tps']})  "
          f"avg_peak_memory_gb={summary['avg_peak_memory_gb']}  "
          f"avg_perplexity={summary['avg_perplexity']}  "
          f"avg_ttft_ms={summary['avg_ttft_ms']}  "
          f"sanity={summary['sanity_check']}  "
          f"quality_pass={summary['quality_pass']}")

    return summary


if __name__ == "__main__":
    diff_from_baseline()
    from inference import generate
    result = evaluate(generate)
    print(json.dumps(result, indent=2))