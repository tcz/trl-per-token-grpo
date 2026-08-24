"""Sequence-level vs per-token advantages, side by side. No GPU, no model.

The task: the model emits a comma-separated list of hex colours, and a checker
scores each colour independently by distance from a target. That is a reward
that *decomposes* — exactly the situation this package exists for.

Both advantage schemes are computed from the same scores, so the difference you
see is purely the credit assignment:

  * sequence-level (what GRPOTrainer does today): collapse each completion to
    one number, normalise across the group, give every token that number.
  * per-token (this package): keep the per-colour scores, map them onto the
    characters that produced them, normalise across the group's token pool.

Run: python examples/span_rewards.py
"""

import re

import torch

from trl_per_token_grpo import (
    Span,
    char_scores_to_token_rewards,
    compute_per_token_advantages,
    span_scores_to_char_scores,
)

TARGET = (0x33, 0x66, 0x99)


def score_colour(hex_text: str) -> float:
    """1.0 for an exact match, decaying with channel distance."""
    try:
        r, g, b = (int(hex_text[i : i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return 0.0
    dist = sum(abs(a - t) for a, t in zip((r, g, b), TARGET)) / (3 * 255)
    return max(0.0, 1.0 - dist)


def spans_for(completion: str) -> list[Span]:
    """One scored span per colour literal in the completion."""
    return [
        Span(m.start(), m.end(), score_colour(m.group()))
        for m in re.finditer(r"#[0-9a-fA-F]{6}", completion)
    ]


def fake_tokenise(text: str) -> list[tuple[int, int]]:
    """Stand-in for a fast tokenizer's offset_mapping: 3-character tokens."""
    return [(i, min(i + 3, len(text))) for i in range(0, len(text), 3)]


def sequence_advantages(rewards: list[float]) -> torch.Tensor:
    """What TRL computes today: one advantage per completion.

    Mirrors ``GRPOTrainer`` with the default ``scale_rewards="group"``:
    ``(r - mean) / (std + 1e-4)``, std with Bessel's correction, over the
    completions of one prompt group.
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(correction=1) + 1e-4)


def main() -> None:
    # Two completions sampled for the same prompt (num_generations=2).
    completions = [
        "#336699, #ff0000",   # first colour exact, second badly wrong
        "#2a5588, #3a6a9a",   # both close, neither exact
    ]

    seq_rewards: list[float] = []
    token_rewards: list[list[float]] = []
    lengths: list[int] = []
    all_offsets: list[list[tuple[int, int]]] = []

    print("scores")
    for text in completions:
        spans = spans_for(text)
        # The completion-level reward a normal reward function would return:
        # the mean of the item scores. Also the natural neutral value for
        # characters no span covers (commas, spaces) — they carry no evidence.
        completion_score = sum(s.score for s in spans) / max(len(spans), 1)

        char_scores = span_scores_to_char_scores(
            len(text), spans, default=completion_score
        )
        offsets = fake_tokenise(text)

        seq_rewards.append(completion_score)
        token_rewards.append(char_scores_to_token_rewards(char_scores, offsets))
        lengths.append(len(offsets))
        all_offsets.append(offsets)

        items = "  ".join(f"{text[s.start:s.end]}={s.score:.3f}" for s in spans)
        print(f"  {text!r}")
        print(f"    {items}   -> completion reward {completion_score:.3f}")

    seq_adv = sequence_advantages(seq_rewards)
    tok_adv = compute_per_token_advantages(
        token_rewards, lengths, max(lengths), group_size=len(completions)
    )

    print("\nadvantages")
    print(f"  {'token':<10}{'sequence-level':>16}{'per-token':>12}   ")
    for i, text in enumerate(completions):
        print(f"  completion {i}: {text!r}")
        for t, (start, end) in enumerate(all_offsets[i]):
            s = seq_adv[i].item()
            p = tok_adv[i, t].item()
            flag = "  <-- opposite sign" if (s < 0) != (p < 0) else ""
            print(f"    {text[start:end]!r:<10}{s:>14.2f}{p:>12.2f}{flag}")

    # Locate the sharpest disagreement to make the point concrete.
    worst_i = worst_t = 0
    worst_gap = 0.0
    for i in range(len(completions)):
        for t in range(lengths[i]):
            gap = tok_adv[i, t].item() - seq_adv[i].item()
            if abs(gap) > abs(worst_gap):
                worst_gap, worst_i, worst_t = gap, i, t
    s0, e0 = all_offsets[worst_i][worst_t]
    frag = completions[worst_i][s0:e0]

    print(
        f"\nSequence-level gives every token in completion 0 the same {seq_adv[0].item():+.2f},\n"
        "including the three tokens that spelled the *exact* target colour — they are\n"
        "punished for sharing a completion with a bad one. Per-token separates them:\n"
        f"the tokens of '#336699' earn {tok_adv[0, 0].item():+.2f} while the tokens of\n"
        f"'#ff0000' earn {tok_adv[0, 3].item():+.2f}.\n\n"
        f"Largest single disagreement: token {frag!r} in completion {worst_i}, "
        f"{seq_adv[worst_i].item():+.2f} -> {tok_adv[worst_i, worst_t].item():+.2f} "
        f"({worst_gap:+.2f}).\n\n"
    )


if __name__ == "__main__":
    main()
