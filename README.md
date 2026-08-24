# trl-per-token-grpo

Per-token advantages for TRL's `GRPOTrainer`.

TRL's loss deliberately *accepts* `(B, T)` advantages — `_compute_loss`
unsqueezes only when the tensor is 1-D, and the comment there names
`trl.experimental.minillm` as the in-tree subclass that supplies them. But
MiniLLM builds its per-token advantages from *discounted return-to-go over
teacher-KL rewards*, which is specific to distillation. There is no general
path for externally scored rewards — a checker, a renderer, a verifier that
scores pieces of the output and hands you back per-span numbers.

This package is that path: token-level group-relative normalisation,
span -> character -> token reward mapping, offset round-trip verification, and a
trainer subclass with a single method to implement.

## Why

Standard GRPO gives every token in a completion the same advantage. If your
reward decomposes — regions of a rendered image, clauses of a proof, cells of a
table, items in a list — that throws away the decomposition. A completion where
one element is perfect and another is broken gets one number, and the tokens
that produced the good element are punished alongside the bad.

Consider the output of `examples/span_rewards.py` where we are fine-tuning
a hypothetical model to match hexadecimal ground-truth colors. 

```
scores
  '#336699, #ff0000'
    #336699=1.000  #ff0000=0.400   -> completion reward 0.700
  '#2a5588, #3a6a9a'
    #2a5588=0.944  #3a6a9a=0.984   -> completion reward 0.964

advantages
  token       sequence-level   per-token   
  completion 0: '#336699, #ff0000'
    '#33'              -0.71        0.75  <-- opposite sign
    '669'              -0.71        0.75  <-- opposite sign
    '9, '              -0.71       -0.07
    '#ff'              -0.71       -1.69
    '000'              -0.71       -1.69
    '0'                -0.71       -1.69
  completion 1: '#2a5588, #3a6a9a'
    '#2a'               0.71        0.52
    '558'               0.71        0.52
    '8, '               0.71        0.57
    '#3a'               0.71        0.68
    '6a9'               0.71        0.68
    'a'                 0.71        0.68

Sequence-level gives every token in completion 0 the same -0.71,
including the three tokens that spelled the *exact* target colour — they are
punished for sharing a completion with a bad one. Per-token separates them:
the tokens of '#336699' earn +0.75 while the tokens of
'#ff0000' earn -1.69.

Largest single disagreement: token '#33' in completion 0, -0.71 -> +0.75 (+1.45).
```

## Install

```bash
pip install -e ".[trainer]"      # with trl, for training
pip install -e .                 # reward-shaping helpers only (torch only)
```

### TRL versions

Requires **trl >= 0.29**. Earlier versions unconditionally unsqueeze the
advantage tensor, so 2-D advantages would be silently mis-shaped; the trainer
raises at construction rather than let that happen.

| | status                                                            |
|---|-------------------------------------------------------------------|
| 0.29 | the version the underlying code was developed and trained against |
| 1.10 (current) | API contract compatible, not tested |

## Use

Implement one method:

```python
from trl_per_token_grpo import (
    PerTokenGRPOTrainer, Span,
    span_scores_to_char_scores, char_scores_to_token_rewards,
)

class MyTrainer(PerTokenGRPOTrainer):
    def compute_token_rewards(self, texts, token_offsets, inputs):
        out = []
        for text, offsets, row in zip(texts, token_offsets, inputs):
            spans = my_scorer(text, row["answer"])
            if spans is None:
                out.append(None)
                continue
            page_score = sum(s.score for s in spans) / len(spans)
            chars = span_scores_to_char_scores(len(text), spans, default=page_score)
            out.append(char_scores_to_token_rewards(chars, offsets))
        return out
```

# License

Apache 2.0