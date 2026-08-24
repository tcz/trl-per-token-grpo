"""Token-level group-relative advantage normalisation."""

from __future__ import annotations

import torch


def compute_per_token_advantages(
    token_rewards: list[list[float]],
    completion_lengths: list[int],
    max_len: int,
    group_size: int | None = None,
    valid: list[bool] | None = None,
) -> torch.Tensor:
    """Normalise per-token rewards into per-token advantages, group-relatively.

    Standard GRPO computes one advantage per completion by normalising rewards
    across the group of completions sampled for a prompt. This does the same
    thing one level down: it pools every *token* reward in a prompt group,
    normalises against that pool, and returns an advantage per token.

    Args:
        token_rewards: ``B`` lists of per-token reward floats.
        completion_lengths: ``B`` unpadded completion lengths.
        max_len: padded width of the returned tensor (``T``).
        group_size: completions per prompt, i.e. ``num_generations``. The batch
            is normalised in consecutive chunks of this size, matching TRL's
            ``RepeatSampler`` layout.
        valid: ``B`` flags. Completions marked invalid are excluded from the
            group statistics and receive advantage 0 everywhere. Use this for
            completions whose reward could not be computed.

    Returns:
        ``(B, max_len)`` float tensor. Padding positions are 0.
    """
    batch = len(token_rewards)
    if valid is None:
        valid = [True] * batch
    if group_size is None or group_size <= 0:
        group_size = batch
    if batch % group_size != 0:
        # Defensive: one group beats silently misaligned groups.
        group_size = batch

    advantages = torch.zeros(batch, max_len, dtype=torch.float32)

    for start in range(0, batch, group_size):
        idxs = range(start, min(start + group_size, batch))

        pool: list[float] = []
        for k in idxs:
            if not valid[k]:
                continue
            length = min(completion_lengths[k], len(token_rewards[k]))
            pool.extend(token_rewards[k][:length])

        if not pool:
            continue

        rewards = torch.tensor(pool, dtype=torch.float32)
        mu = rewards.mean()
        # Population std: the group's token pool *is* the population for this
        # prompt, not a sample drawn from a larger one.
        sigma = rewards.std(correction=0)

        if sigma < 1e-8:
            continue  # no variance in this group; advantages stay 0

        for k in idxs:
            if not valid[k]:
                continue
            length = min(completion_lengths[k], len(token_rewards[k]))
            for t in range(length):
                advantages[k, t] = (token_rewards[k][t] - mu) / sigma

    return advantages
