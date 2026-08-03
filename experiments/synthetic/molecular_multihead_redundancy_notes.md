# Multi-head redundant routing: local experiment notes

## Question

Can a small transformer with no explicit local/global mixture learn redundant
dense routing, and can clean semantic--structural conditional carriage recover
how much of that route the model has instantiated without claiming that the
route is task-necessary?

The intended estimand is **model-internal route allocation**.  Local-route
failure rescue and distant-route failure damage are behavioural validation
controls.  Neither is interpreted as task necessity.

## Minimal setup

- Five tokens on real ZINC molecular supports: a local anchor and four records
  at shortest-path distances `[1, 4, 4, 6]`.
- The anchor contains a redundant copy of the target.
- A semantic bit selects the record key and a structural bit selects its bank.
- Two dense transformer layers, four heads, hidden dimension 32.
- No explicit local/global gate.  Route preference is induced only by the
  relative sign-flip rates of the local and selected distant copies.
- Final-state semantic, structural, and 2x2 conditional-interaction carriage
  are measured on clean held-out examples.  Head attention responses are
  measured separately.

## Main four-seed result

The summed-token graph readout is the valid distance-resolved design.  Values
below are means over four independently trained seeds.

| Local train corruption | Clean MAE | Interaction mass | Interaction expected distance | Local-failure rescue | Distant-failure damage | Head score alignment | Relative head imbalance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.022 | 0.020 | 2.56 | 0.008 | 0.003 | 0.771 | 0.288 |
| 0.01 | 0.050 | 0.115 | 2.06 | 0.119 | 0.089 | 0.822 | 0.282 |
| 0.05 | 0.134 | 0.428 | 2.43 | 0.556 | 0.443 | 0.899 | 0.214 |
| 0.20 | 0.107 | 1.382 | 2.50 | 1.643 | 1.533 | 0.939 | 0.183 |

Across the 16 trained models:

- interaction mass versus local-failure rescue: `r = 0.992`;
- interaction mass versus distant-failure damage: `r = 0.994`;
- training local corruption versus interaction mass: `r = 0.973`;
- mean attention interaction versus local-failure rescue: `r = 0.934`.

The raw interaction mass changes strongly while its normalized expected
distance remains near the same fixed carrier geometry.  Here the amount of
conditional carriage is informative; normalized distance alone is not.

## What worked

1. **The effect emerges in an ordinary multi-head transformer.**  The strong
   carriage--behaviour relationship from the explicit-mixture control survives
   after removing the mixture parameter.
2. **The measure has a two-sided behavioural interpretation.**  More clean
   interaction carriage predicts both better rescue when the local copy fails
   and greater damage when the distant selected record fails.  It therefore
   measures instantiated backup capacity and the matching exposure, not
   necessity.
3. **Semantic and structural score co-peaking has an explainable positive
   case.**  As the conjunctive distant lookup is used more, the semantic and
   structural head-score vectors align (`0.771 -> 0.939`) and their
   scale-normalized imbalance falls (`0.288 -> 0.183`).  Both factors address
   the same record, so co-peaking can represent conjunctive routing rather than
   a division into semantic-only and structural-only heads.
4. **The aggregate is more stable than head identity.**  The maximum-response
   head changes across seeds and head indices.  The defensible claim is about
   distributed model organization, not a necessary named specialist.

## Iterations and negative results

### More-data control

The initial 512-example run underfit the intermediate reliability condition.
A two-seed control with 2,048 training examples and otherwise unchanged
methodology reduced clean MAE and produced the expected route allocation:

| Local / distant corruption | Clean MAE | Interaction mass | Rescue | Damage | Head alignment |
|---:|---:|---:|---:|---:|---:|
| 0.05 / 0.05 | 0.064 | 1.083 | 1.005 | 0.970 | 0.938 |
| 0.20 / 0.05 | 0.091 | 1.450 | 1.603 | 1.562 | 0.969 |

Equal route reliability therefore approaches symmetric behavioural reliance.
The smaller-run asymmetry should not be interpreted scientifically.

### Anchor-only readout

An anchor-only readout was tested as an optimization control.  It made the
easy clean condition more accurate, but every final-state output gradient then
lives at the anchor.  All carriage expected distances became exactly zero even
though attention still routed to distant records.  This is a readout-induced
degeneracy, so the anchor result is rejected for distance-resolved carriage and
retained only as a sensitivity control.

### Raw division-of-labour score

The unnormalized mean absolute semantic--structural difference grows with
overall response scale, even while relative imbalance falls and score vectors
align.  It is not interpretable across route-use regimes without a scale-aware
companion.

## Boundaries

- The failure tests establish reliance on a redundant route, not minimal task
  necessity.  A local-only model class can solve the clean task exactly.
- The record-bank task is deliberately simple and aligned with the measured
  semantic/structural interventions.  The high correlations are a positive
  control, not a promised real-world effect size.
- Fixed record geometry makes this a test of route amount at known distances,
  not a test of learning a longer spatial reach.
- Head attention responses are an interpretable routing companion, not a
  substitute for the dissertation's output-linked specialization scores.

## Decision

This synthetic is strong enough as a validation bridge:

1. the matched additive/conjunctive softmax control identifies what the 2x2
   interaction term distinguishes;
2. this learned multi-head control shows the signal can emerge without an
   explicit gate and has redundant-route behavioural meaning;
3. the ZINC pilot can now ask whether the same aggregate relationship is
   present, weak, or absent in a real trained model without making a necessity
   claim.

The clean chapter claim should be that conditional carriage distinguishes the
**amount and organization of semantic--structural dense routing used by the
trained model**, while separate sparse-model performance establishes that such
routing need not be required by the task.

## Reproducible artifacts

- Main implementation: `src/graph_specialisation_metrics/synthetic/molecular_multihead_redundancy.py`
- Four-seed result: `outputs/molecular_multihead_redundancy_sum_v2/results.json`
- Larger-data control: `outputs/molecular_multihead_redundancy_data_control/results.json`
- Rejected anchor control: `outputs/molecular_multihead_redundancy_anchor/results.json`
