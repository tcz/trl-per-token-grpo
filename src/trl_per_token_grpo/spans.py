"""Mapping span-level scores onto characters, then onto tokens."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A scored character range in the completion.

    Attributes:
        start: inclusive character offset.
        end: exclusive character offset.
        score: quality of this span, on whatever scale ``blend`` expects.
        weight: relative importance when spans overlap (e.g. area, token count).
            Overlapping spans are combined as a weight-weighted mean.
        innermost_wins: if True, this span belongs to a nesting hierarchy and
            the *smallest* covering span takes a character outright rather than
            being averaged with its parents. Use for tree-structured spans
            (HTML elements); leave False for flat, possibly-overlapping ones
            (style rules matching several elements).
    """

    start: int
    end: int
    score: float
    weight: float = 1.0
    innermost_wins: bool = False


def span_scores_to_char_scores(
    text_len: int,
    spans: list[Span],
    default: float,
) -> list[float]:
    """Resolve possibly-overlapping spans into one score per character.

    Characters covered by no span get ``default``. Choose ``default`` so that
    unmapped characters are *neutral* on the same scale as mapped ones — the
    completion-level score is usually right. A default that is systematically
    better or worse than typical mapped scores means the model is rewarded for
    producing text your reward cannot see, which is rarely what you want and is
    easy to do by accident.

    Args:
        text_len: length of the completion text.
        spans: scored ranges; may overlap.
        default: score for uncovered characters.

    Returns:
        ``text_len`` floats.
    """
    if text_len <= 0:
        return []

    # (score, weight) contributions per character
    flat: list[list[tuple[float, float]]] = [[] for _ in range(text_len)]
    # (score, weight, span_width) candidates for hierarchical spans
    nested: list[list[tuple[float, float, int]]] = [[] for _ in range(text_len)]

    for span in spans:
        lo = max(0, span.start)
        hi = min(text_len, span.end)
        if hi <= lo:
            continue
        if span.innermost_wins:
            width = span.end - span.start
            for c in range(lo, hi):
                nested[c].append((span.score, span.weight, width))
        else:
            for c in range(lo, hi):
                flat[c].append((span.score, span.weight))

    # Smallest covering hierarchical span wins outright, then joins the pool.
    for c in range(text_len):
        if nested[c]:
            best = min(nested[c], key=lambda x: x[2])
            flat[c].append((best[0], best[1]))

    out: list[float] = []
    for c in range(text_len):
        contributions = flat[c]
        if not contributions:
            out.append(default)
            continue
        total_w = sum(w for _, w in contributions)
        if total_w < 1e-12:
            out.append(sum(s for s, _ in contributions) / len(contributions))
        else:
            out.append(sum(s * w for s, w in contributions) / total_w)
    return out


def char_scores_to_token_rewards(
    char_scores: list[float],
    token_offsets: list[tuple[int, int]],
) -> list[float]:
    """Average character scores over each token's character span.

    ``token_offsets`` comes from a fast tokenizer:
    ``tok(text, return_offsets_mapping=True)["offset_mapping"]``.

    Empty spans (``start == end``, typical for special tokens) receive the mean
    of all character scores, which keeps them neutral.
    """
    if not char_scores:
        return [0.0] * len(token_offsets)

    global_mean = sum(char_scores) / len(char_scores)

    rewards: list[float] = []
    for start, end in token_offsets:
        if start >= end:
            rewards.append(global_mean)
        else:
            window = char_scores[start:end]
            rewards.append(sum(window) / len(window) if window else global_mean)
    return rewards
