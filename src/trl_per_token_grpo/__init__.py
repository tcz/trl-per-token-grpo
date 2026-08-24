"""Per-token advantages for TRL's GRPOTrainer."""

from .advantages import compute_per_token_advantages
from .spans import Span, char_scores_to_token_rewards, span_scores_to_char_scores
from .verify import pad_to_length, round_trip_ok

__all__ = [
    "compute_per_token_advantages",
    "Span",
    "span_scores_to_char_scores",
    "char_scores_to_token_rewards",
    "round_trip_ok",
    "pad_to_length",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Trainer classes import trl, which is heavy and not needed for the pure
    # reward-shaping helpers. Load them lazily so `import trl_per_token_grpo`
    # stays cheap and works without trl installed.
    if name in {"PerTokenGRPOTrainer", "ChunkedLogpsMixin"}:
        from . import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
