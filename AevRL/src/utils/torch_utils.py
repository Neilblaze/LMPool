import torch

from ..utils.config import RolloutConfig


def get_masks_from_tokens(tokens: list, config: RolloutConfig):
    """Label tokens between <im_start>assistant and <im_end> as 1, leave everything else as 0"""
    if not tokens or not config.mask_start_token_ids or not config.mask_end_token_ids:
        return []

    mask = [0] * len(tokens)
    start_len = len(config.mask_start_token_ids)
    end_len = len(config.mask_end_token_ids)
    if len(tokens) < min(start_len, end_len):
        return mask

    in_mask = False
    idx = 0
    while idx < len(tokens):
        if tokens[idx : idx + start_len] == config.mask_start_token_ids:
            in_mask = True
            idx += start_len
            continue
        if tokens[idx : idx + end_len] == config.mask_end_token_ids:
            in_mask = False
            idx += end_len
            continue
        if in_mask:
            mask[idx] = 1
        idx += 1

    return mask


def move_opt_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)


def pad_2d(
    seqs: list[list[int | float]], pad_value: int | float, dtype: torch.dtype, device: str
) -> torch.Tensor:
    max_len = max((len(seq) for seq in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_value, dtype=dtype, device=device)
    for row_idx, seq in enumerate(seqs):
        out[row_idx, : len(seq)] = torch.tensor(seq, dtype=dtype, device=device)
    return out
