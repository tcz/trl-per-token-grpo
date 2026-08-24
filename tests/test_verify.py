from trl_per_token_grpo import pad_to_length, round_trip_ok


def test_identical_round_trip_passes():
    assert round_trip_ok([1, 2, 3, 4], [1, 2, 3, 4])


def test_dropped_trailing_eos_is_tolerated():
    assert round_trip_ok([1, 2, 3, 99], [1, 2, 3])


def test_divergence_is_caught():
    assert not round_trip_ok([1, 2, 3, 4], [1, 2, 9, 4])


def test_shift_is_caught():
    """The failure that matters: every reward after the shift lands wrong."""
    assert not round_trip_ok([1, 2, 3, 4], [2, 3, 4])


def test_too_short_is_caught():
    assert not round_trip_ok([1, 2, 3, 4], [1, 2])


def test_longer_reencoding_is_caught():
    assert not round_trip_ok([1, 2, 3], [1, 2, 3, 4])


def test_pad_uses_own_mean():
    padded = pad_to_length([0.2, 0.4], 3)
    assert len(padded) == 3
    assert abs(padded[2] - 0.3) < 1e-9


def test_pad_truncates_when_too_long():
    assert pad_to_length([0.1, 0.2, 0.3], 2) == [0.1, 0.2]


def test_pad_of_empty_list_is_zeros():
    assert pad_to_length([], 2) == [0.0, 0.0]
