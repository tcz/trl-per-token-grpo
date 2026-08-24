"""GRPOTrainer subclasses that train on per-token advantages."""

from __future__ import annotations

import os
import warnings
from typing import Sequence

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

from .advantages import compute_per_token_advantages
from .verify import pad_to_length, round_trip_ok


class PerTokenGRPOTrainer(GRPOTrainer):
    """GRPO with one advantage per token instead of one per completion.

    TRL's ``_compute_loss`` accepts ``(B, T)`` advantages. It
    conditionally unsqueezes only when the tensor is 1-D, for subclasses that
    supply them. MiniLLM derives its per-token advantages from discounted
    return-to-go over teacher-KL rewards, though; there is no general path
    for rewards that come from an external scorer over spans of the output.

    Implement `compute_token_rewards`.

    Example::

        class MyTrainer(PerTokenGRPOTrainer):
            def compute_token_rewards(self, texts, offsets, inputs):
                out = []
                for text, offs in zip(texts, offsets):
                    char = my_scoring_function(text)          # list[float]
                    out.append(char_scores_to_token_rewards(char, offs))
                return out
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._assert_trl_supports_2d_advantages()

    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    def compute_token_rewards(
        self,
        completions_text: list[str],
        token_offsets: list[list[tuple[int, int]]],
        inputs: Sequence[dict],
    ) -> list[list[float] | None]:
        """Return one reward per token for each completion.

        Args:
            completions_text: decoded completions (special tokens skipped).
            token_offsets: per completion, the ``(start, end)`` character span
                of each token, already verified to align with the generated
                ids. Same length as the token sequence.
            inputs: the batch as TRL assembled it, repeated ``num_generations``
                times per prompt — use it to reach ground truth, images, etc.

        Returns:
            One list of per-token rewards per completion, or ``None`` for a
            completion you could not score. ``None`` marks it invalid: it is
            excluded from the group statistics and trained with advantage 0,
            rather than being assigned a made-up reward.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_trl_supports_2d_advantages() -> None:
        import inspect

        try:
            source = inspect.getsource(GRPOTrainer._compute_loss)
        except (OSError, TypeError):  # pragma: no cover - source unavailable
            return
        if "dim() == 1" not in source:
            import trl

            raise RuntimeError(
                f"trl {getattr(trl, '__version__', '?')} unconditionally "
                "unsqueezes the advantage tensor, so (B, T) advantages would be "
                "silently mis-shaped. Upgrade to trl>=0.29, or override "
                "_compute_loss yourself."
            )

    def _generate_and_score_completions(self, inputs, **kwargs):
        output = super()._generate_and_score_completions(inputs, **kwargs)

        completion_ids = output["completion_ids"]      # (B, T)
        completion_mask = output["completion_mask"]    # (B, T)
        batch, width = completion_ids.shape
        lengths = [int(v) for v in completion_mask.sum(dim=1).tolist()]

        tokenizer = getattr(self.processing_class, "tokenizer", self.processing_class)
        texts = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        offsets: list[list[tuple[int, int]]] = []
        valid: list[bool] = []
        n_mismatch = 0

        for i in range(batch):
            text = texts[i]
            if not text.strip():
                offsets.append([])
                valid.append(False)
                continue
            try:
                enc = tokenizer(
                    text, return_offsets_mapping=True, add_special_tokens=False
                )
            except Exception as exc:  # pragma: no cover - tokenizer-specific
                warnings.warn(f"tokenisation failed for completion {i}: {exc}")
                offsets.append([])
                valid.append(False)
                continue

            actual = completion_ids[i, : lengths[i]].tolist()
            if not round_trip_ok(actual, list(enc["input_ids"])):
                n_mismatch += 1
                offsets.append([])
                valid.append(False)
                continue

            offsets.append([tuple(o) for o in enc["offset_mapping"]])
            valid.append(True)

        scorable = [i for i in range(batch) if valid[i]]
        rewards_by_index: dict[int, list[float]] = {}
        if scorable:
            produced = self.compute_token_rewards(
                [texts[i] for i in scorable],
                [offsets[i] for i in scorable],
                [inputs[i] for i in scorable] if inputs is not None else [],
            )
            if len(produced) != len(scorable):
                raise ValueError(
                    "compute_token_rewards returned "
                    f"{len(produced)} entries for {len(scorable)} completions"
                )
            for slot, i in enumerate(scorable):
                entry = produced[slot]
                if entry is None:
                    valid[i] = False
                else:
                    rewards_by_index[i] = list(entry)

        token_rewards = [
            pad_to_length(rewards_by_index.get(i, []), lengths[i])
            if valid[i]
            else [0.0] * lengths[i]
            for i in range(batch)
        ]

        advantages = compute_per_token_advantages(
            token_rewards,
            lengths,
            width,
            group_size=self.num_generations,
            valid=valid,
        )
        output["advantages"] = advantages.to(output["advantages"].device)

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["per_token/valid_ratio"].append(
            sum(valid) / max(batch, 1)
        )
        self._metrics[mode]["per_token/offset_mismatch_ratio"].append(
            n_mismatch / max(batch, 1)
        )
        return output


class ChunkedLogpsMixin:
    """Force log-prob forwards to run in fixed-size chunks.

    TRL chunks the log-prob forward by ``per_device_{train,eval}_batch_size``.
    The intermediate logits tensor is ``chunk x seq_len x vocab`` in fp32,
    which for long completions and a large vocabulary is tens of gigabytes —
    tolerable on a dedicated training GPU, fatal when vLLM shares the card in
    colocate mode. Chunking trades a few sequential forwards for bounded peak
    memory.

    Set the chunk size with ``PER_TOKEN_LOGPS_CHUNK`` (default 1). Mix in
    before the trainer class::

        class MyTrainer(ChunkedLogpsMixin, PerTokenGRPOTrainer):
            ...
    """

    logps_chunk_size: int = int(os.environ.get("PER_TOKEN_LOGPS_CHUNK", "1"))

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep,
        batch_size=None, **kwargs,
    ):
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            batch_size=self.logps_chunk_size, **kwargs,
        )
