import os

import requests

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")


def _tokenize_messages(model: str, prompt: list[dict]) -> list[int]:
    resp = requests.post(
        f"{VLLM_BASE_URL}/tokenize",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": prompt,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return list(data["tokens"])


def _sleep(level: int = 1):
    resp = requests.post(f"{VLLM_BASE_URL}/sleep?level={level}", timeout=30)
    resp.raise_for_status()


def _wake_up(tags: str | None = None) -> None:
    resp = requests.post(f"{VLLM_BASE_URL}/wake_up{f'?tags={tags}' if tags else ''}", timeout=30)
    resp.raise_for_status()


def adapter_exists(adapter_path: str) -> bool:
    adapter_config = os.path.join(os.path.abspath(adapter_path), "adapter_config.json")
    return os.path.exists(adapter_config)


def _unload_lora(adapter_name: str):
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/unload_lora_adapter",
        headers={"Content-Type": "application/json"},
        json={
            "lora_name": adapter_name,
        },
        timeout=60,
    )
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return resp.status_code == 200


def _load_lora(adapter_name: str, adapter_path: str):
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/load_lora_adapter",
        headers={"Content-Type": "application/json"},
        json={"lora_name": adapter_name, "lora_path": adapter_path},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.status_code == 200


def _reload_with_lora(
    adapter_name: str,
    adapter_path: str,
) -> bool:
    _wake_up()

    adapter_path = os.path.abspath(adapter_path)
    if not adapter_exists(adapter_path):
        return False

    _unload_lora(adapter_name)
    loaded = _load_lora(adapter_name, adapter_path)
    return loaded


def _ping():
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/ping")
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise Exception("vLLM not running.") from exc


def _get_model_logps(
    model: str,
    prompt: list[int] | str,
    return_token_ids: bool = True,
) -> list[tuple[int, float]] | list[float]:
    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": prompt,  # could be string or list of int tokens
            "max_tokens": 0,  # strictly to get logps of whole run
            "echo": True,
            "logprobs": 1,
            "prompt_logprobs": 1,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    token_logprobs = choice["logprobs"]["token_logprobs"]
    logprobs = [0.0 if lp is None else float(lp) for lp in token_logprobs]

    if not return_token_ids:
        return logprobs

    if isinstance(prompt, list):
        token_ids = list(prompt)
    else:
        prompt_logprobs = choice["prompt_logprobs"]
        token_ids = []
        for token_info in prompt_logprobs:
            if token_info is None:
                token_ids.append(0)
                continue

            token_id, _ = next(iter(token_info.items()))
            token_ids.append(int(token_id))

    return list(zip(token_ids, logprobs))
