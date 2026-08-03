# Two overlapping clues: stress-test notes

## Simple question

Does semantic-structural interaction carriage still distinguish model usage
from task benefit when the local and distant routes are imperfect, overlapping
clues rather than exact copies of the target?

## Task

- The target is a hidden number.
- The anchor contains a noisy local clue.
- One of four records contains a second noisy clue. A semantic key and
  structural bank jointly identify that record.
- The records occupy distances `[1, 4, 4, 6]` sampled from real ZINC graph
  supports.
- The clues have equal marginal quality. A controlled fraction of their error
  is shared, so they make increasingly similar mistakes.
- A local-only linear fit and a fit using both known clues measure the maximum
  clean benefit available from the distant clue.
- A two-layer, four-head transformer must learn record selection; it has no
  explicit local/distant mixture parameter.

This replaces an exact duplicated target with a more realistic regression
setting in which redundant signals can be useful, weakly useful, or entirely
duplicative.

## Iteration 1: ordinary training

Without clue dropout, the transformer behaves economically. When the clues
have independent errors, it learns the distant lookup. When their errors are
fully shared, it uses the easier local route and mostly abandons the distant
one.

| Shared error | Clean gain available from two clues | Interaction mass | Distant-change damage |
|---:|---:|---:|---:|
| 0% | 0.088 | 0.636 | 0.444 |
| 100% | 0.000 | 0.049 | 0.004 |

This is a useful negative control: interaction carriage responds to actual
route use, not merely to a distant clue being present in the input.

## Iteration 2: occasional clue hiding during training

Each clue was independently set to zero in 25% of training examples. This is a
simple robustness pressure: sometimes the model must use the other clue. Clean
evaluation still contains both clues.

Four-seed means:

| Shared error | Available clean gain | Model gain over local fit | Interaction mass | Distant-change damage | Exact output interaction | Head-score alignment |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.088 | 0.062 | 0.842 | 0.376 | 0.757 | 0.977 |
| 50% | 0.047 | 0.027 | 0.643 | 0.229 | 0.579 | 0.963 |
| 90% | 0.010 | -0.013 | 0.652 | 0.135 | 0.562 | 0.988 |
| 100% | 0.000 | -0.008 | 0.565 | 0.091 | 0.509 | 0.965 |

At 100% shared error, the local and distant clues are numerically identical on
every clean example. A two-clue fit therefore has exactly the same error as a
local-only fit. Nevertheless, all four transformer seeds retain non-zero
distant-change damage and substantial interaction carriage:

- interaction mass by seed: `0.431, 0.645, 0.682, 0.503`;
- distant-change damage by seed: `0.034, 0.104, 0.151, 0.076`.

Compared with ordinary training at the same 100% overlap, clue hiding produces
about 11.5 times more interaction mass and 21 times more distant-change damage.
The distant route is used because of the training pressure, not because it
adds unique information on clean examples.

## What survives

1. **Interaction carriage distinguishes availability from use.** A distant
   clue can be present but ignored under ordinary training.
2. **Use is not unique benefit.** Robustness training can create a used distant
   route even when its clean-data benefit is exactly zero.
3. **The behavioural relationship survives but weakens realistically.** Across
   the 16 robustness-trained models, interaction mass predicts damage from
   changing the distant record with `r = 0.850`, rather than the near-perfect
   relationship in the exact-copy calibration.
4. **Score co-peaking is about organization, not necessity.** At full overlap,
   robustness-trained models have strongly aligned semantic and structural
   head profiles (`0.965`) while the distant clue has zero unique clean value.
   The same conjunctive route requires both selectors, but it is not necessary
   for the clean task.
5. **Raw amount remains more informative than normalised distance.** Mean
   interaction distance stays near `2.1-2.2` while route mass and behavioural
   reliance change substantially.

## What does not survive as a simple claim

- Interaction mass is not a measure of how much the route improves clean task
  performance. It measures how much of the route the trained model uses.
- Reliance is no longer an exact linear function of interaction mass once the
  clues are noisy and correlated. The relationship remains strong but includes
  model and seed variation.
- Dense-route use cannot be explained from the data distribution alone.
  Training regularisation changes which equivalent solution is learned.

## Dissertation use

This is a stronger stress test than the exact-copy task because it supplies a
clean counterexample to the tempting interpretation "more carriage means more
unique task information." The defensible statement is narrower and more useful:

> Factorial carriage measures the amount and semantic-structural organization
> of a route used by the trained model. Separate restricted-model performance
> is required to judge whether that route adds unique task value.

The experiment remains controlled: records are sparse registered tokens, and
zeroing a clue is simpler than molecular feature corruption. It should support
the interpretation of the real analysis, not stand in for it.

## Artifacts

- Implementation: `src/graph_specialisation_metrics/synthetic/molecular_correlated_clues.py`
- Four-seed robustness run: `outputs/molecular_correlated_clues_dropout25/results.json`
- Matched no-dropout endpoints: `outputs/molecular_correlated_clues_no_dropout_control/results.json`
- Tests: `tests/test_molecular_correlated_clues.py`
