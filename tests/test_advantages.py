import torch

from trl_per_token_grpo import compute_per_token_advantages


def test_high_reward_tokens_get_positive_advantage():
    adv = compute_per_token_advantages([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]], [3, 3], 3)
    assert (adv[0] > 0).all()
    assert (adv[1] < 0).all()
    assert abs(adv.mean().item()) < 1e-5


def test_tokens_disagree_within_one_completion():
    """The whole point: one completion, opposite-signed tokens."""
    adv = compute_per_token_advantages([[0.9, 0.1], [0.5, 0.5]], [2, 2], 2)
    assert adv[0, 0] > 0 > adv[0, 1]


def test_padding_is_zero():
    adv = compute_per_token_advantages([[0.8, 0.7, 0.9], [0.2, 0.3]], [3, 2], 4)
    assert adv[0, 3] == 0.0
    assert adv[1, 2] == 0.0 and adv[1, 3] == 0.0


def test_zero_variance_group_yields_zero_advantages():
    adv = compute_per_token_advantages([[0.5] * 3, [0.5] * 3], [3, 3], 3)
    assert (adv.abs() < 1e-6).all()


def test_group_size_isolates_prompts():
    """Without grouping, an easy prompt's rewards swamp a hard prompt's."""
    rewards = [[0.9, 0.9], [0.7, 0.7], [0.3, 0.3], [0.1, 0.1]]
    lengths = [2, 2, 2, 2]

    grouped = compute_per_token_advantages(rewards, lengths, 2, group_size=2)
    # Within each pair the better completion is positive, the worse negative.
    assert (grouped[0] > 0).all() and (grouped[1] < 0).all()
    assert (grouped[2] > 0).all() and (grouped[3] < 0).all()

    # Pooling the whole batch instead makes the entire second prompt negative.
    pooled = compute_per_token_advantages(rewards, lengths, 2)
    assert (pooled[2] < 0).all()


def test_invalid_completions_are_neutral_and_excluded():
    adv = compute_per_token_advantages(
        [[0.8, 0.8], [0.0, 0.0], [0.2, 0.2]],
        [2, 2, 2],
        2,
        valid=[True, False, True],
    )
    assert (adv[1] == 0).all()                      # neutral, not punished
    assert torch.allclose(adv[0], -adv[2])          # stats from the valid pair
    assert (adv[0] > 0).all()


def test_all_invalid_group_is_all_zero():
    adv = compute_per_token_advantages([[0.0], [0.0]], [1, 1], 1, valid=[False, False])
    assert (adv == 0).all()


def test_mismatched_group_size_falls_back_to_single_group():
    # 3 completions, group_size 2 -> not divisible; must not misalign groups.
    adv = compute_per_token_advantages([[0.9], [0.5], [0.1]], [1, 1, 1], 1, group_size=2)
    assert adv.shape == (3, 1)
    assert adv[0, 0] > adv[1, 0] > adv[2, 0]


class TestAgainstSequenceLevel:
    """The property the package exists for: per-token advantages can flip sign
    relative to the sequence-level advantage their completion would receive."""

    @staticmethod
    def _sequence_advantages(rewards):
        """TRL's formula: (r - mean) / (std + 1e-4), Bessel-corrected std."""
        r = torch.tensor(rewards, dtype=torch.float32)
        return (r - r.mean()) / (r.std(correction=1) + 1e-4)

    def test_good_tokens_in_a_bad_completion_flip_sign(self):
        # Completion 0: one excellent half, one terrible half -> poor overall.
        # Completion 1: uniformly good -> better overall.
        token_rewards = [[1.0, 1.0, 0.4, 0.4], [0.94, 0.94, 0.98, 0.98]]
        seq_rewards = [sum(r) / len(r) for r in token_rewards]

        seq = self._sequence_advantages(seq_rewards)
        tok = compute_per_token_advantages(token_rewards, [4, 4], 4, group_size=2)

        # Sequence-level condemns the whole of completion 0...
        assert seq[0] < 0
        # ...including the tokens that were actually excellent.
        assert tok[0, 0] > 0 and tok[0, 1] > 0     # the good half: rewarded
        assert tok[0, 2] < 0 and tok[0, 3] < 0     # the bad half: punished
        # i.e. the sign genuinely flips for those tokens.
        assert (seq[0] < 0) and (tok[0, 0] > 0)

    def test_uniform_rewards_reduce_to_the_sequence_level_ordering(self):
        """With no within-completion variation, per-token should agree with
        sequence-level on which completion is better."""
        token_rewards = [[0.9, 0.9, 0.9], [0.3, 0.3, 0.3]]
        seq = self._sequence_advantages([0.9, 0.3])
        tok = compute_per_token_advantages(token_rewards, [3, 3], 3, group_size=2)

        assert (seq[0] > 0) == bool((tok[0] > 0).all())
        assert (seq[1] < 0) == bool((tok[1] < 0).all())
