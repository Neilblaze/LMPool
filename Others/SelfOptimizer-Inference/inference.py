import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

MAX_TOKENS = 256
PREFILL_STEP_SIZE = 2048
_SAMPLER = make_sampler(temp=0.0)

_EMPTY = {
    "text": "", "generation_tps": 0.0, "prompt_tps": 0.0,
    "peak_memory_gb": 0.0, "generation_tokens": 0, "prompt_tokens": 0,
}


def generate(model, tokenizer, prompt: str) -> dict:
    # Generate text and return metrics (tps, memory, token counts)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
    )

    output, final = "", None
    for chunk in stream_generate(
        model, tokenizer, prompt_ids,
        max_tokens=MAX_TOKENS, sampler=_SAMPLER, prefill_step_size=PREFILL_STEP_SIZE,
    ):
        output += chunk.text
        final = chunk

    return _EMPTY.copy() if final is None else {
        "text": output,
        "generation_tps": final.generation_tps,
        "prompt_tps": final.prompt_tps,
        "peak_memory_gb": final.peak_memory,
        "generation_tokens": final.generation_tokens,
        "prompt_tokens": final.prompt_tokens,
    }