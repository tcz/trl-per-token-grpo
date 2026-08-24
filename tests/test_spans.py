from trl_per_token_grpo import (
    Span,
    char_scores_to_token_rewards,
    span_scores_to_char_scores,
)


def test_uncovered_characters_get_the_default():
    scores = span_scores_to_char_scores(5, [], default=0.7)
    assert scores == [0.7] * 5


def test_span_scores_land_on_their_characters():
    scores = span_scores_to_char_scores(
        6, [Span(0, 3, 0.9), Span(3, 6, 0.1)], default=0.5
    )
    assert scores[:3] == [0.9] * 3
    assert scores[3:] == [0.1] * 3


def test_overlapping_spans_combine_by_weight():
    scores = span_scores_to_char_scores(
        2, [Span(0, 2, 1.0, weight=3.0), Span(0, 2, 0.0, weight=1.0)], default=0.5
    )
    assert all(abs(s - 0.75) < 1e-9 for s in scores)


def test_innermost_span_wins_over_its_parent():
    """Nested structure: the child owns its characters outright."""
    spans = [
        Span(0, 10, 0.2, weight=100.0, innermost_wins=True),   # parent
        Span(4, 6, 0.9, weight=1.0, innermost_wins=True),      # child
    ]
    scores = span_scores_to_char_scores(10, spans, default=0.0)
    assert scores[4] == 0.9 and scores[5] == 0.9               # child region
    assert scores[0] == 0.2 and scores[9] == 0.2               # parent's own text


def test_zero_weight_spans_fall_back_to_plain_mean():
    scores = span_scores_to_char_scores(
        1, [Span(0, 1, 1.0, weight=0.0), Span(0, 1, 0.0, weight=0.0)], default=0.5
    )
    assert abs(scores[0] - 0.5) < 1e-9


def test_spans_are_clipped_to_the_text():
    scores = span_scores_to_char_scores(3, [Span(-5, 99, 1.0)], default=0.0)
    assert scores == [1.0, 1.0, 1.0]


def test_char_scores_average_over_token_spans():
    rewards = char_scores_to_token_rewards([0.8, 0.8, 0.2, 0.2], [(0, 2), (2, 4)])
    assert abs(rewards[0] - 0.8) < 1e-9
    assert abs(rewards[1] - 0.2) < 1e-9


def test_empty_token_spans_get_the_global_mean():
    rewards = char_scores_to_token_rewards([0.6, 0.4], [(0, 0), (0, 1), (1, 2)])
    assert abs(rewards[0] - 0.5) < 1e-9
    assert abs(rewards[1] - 0.6) < 1e-9
