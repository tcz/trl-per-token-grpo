"""Verifying that character offsets actually line up with generated tokens."""

from __future__ import annotations


def round_trip_ok(
    completion_ids: list[int],
    reencoded_ids: list[int],
    allow_trailing_slack: int = 1,
) -> bool:
    """Check that re-tokenising the decoded text reproduces the generated ids.

    Args:
        completion_ids: the ids actually generated (unpadded).
        reencoded_ids: ids from re-tokenising the decoded text with
            ``add_special_tokens=False``.
        allow_trailing_slack: how many trailing ids may be missing from the
            re-encoding. Decoding with ``skip_special_tokens=True`` normally
            drops a trailing EOS, so 1 is the sensible default.

    Returns:
        True if the re-encoding is a prefix of the generated ids and is at most
        ``allow_trailing_slack`` tokens shorter.
    """
    n_re = len(reencoded_ids)
    n_orig = len(completion_ids)
    if not (n_orig - allow_trailing_slack <= n_re <= n_orig):
        return False
    return reencoded_ids == completion_ids[:n_re]


def pad_to_length(rewards: list[float], length: int) -> list[float]:
    """Pad a token-reward list up to ``length`` with its own mean.

    Only sound for the small gap ``round_trip_ok`` tolerates (the dropped EOS).
    Do not use it to paper over a real mismatch — mark the completion invalid
    instead, so it is excluded from the group statistics rather than trained on
    with misaligned rewards.
    """
    if len(rewards) >= length:
        return rewards[:length]
    filler = sum(rewards) / len(rewards) if rewards else 0.0
    return rewards + [filler] * (length - len(rewards))
