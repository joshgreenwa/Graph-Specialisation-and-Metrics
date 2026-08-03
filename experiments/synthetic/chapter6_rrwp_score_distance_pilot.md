# Chapter 6 synthetic iteration: RRWP performance versus score distance

## Decision

This is a useful conceptual control and supports proceeding to the paired ZINC
comparison. It establishes that a real local/global RRWP performance gap need
not be accompanied by a difference in either confidence-matched raw
specialisation scores or their distance geometry.

The result also clarifies what a future null would mean. Similar score profiles
do not imply that the two RRWP scopes contain equally useful structural
information. They imply only that the fitted models organise their response in
a similar way under the registered interventions.

## Task and controls

Each graph contains one cycle candidate and one chain candidate carrying
independent standard-normal values. The regression target is the value on the
cycle. A leaf attached near one candidate creates a one-step local-pair clue
that identifies the cycle with probability 0.75. Diagonal RRWP through order 16
provides a substantially more reliable clue.

The local and global models are matched linear structural gates with the same
18-dimensional feature vector and parameter count. Higher-order RRWP channels
are zeroed in the local model. Both models use the selected candidate value in
exactly the same way. Eight paired seeds each use 8,192 training and 8,192 test
examples.

Two comparisons are reported:

- **Freely fitted:** each gate retains its learned confidence.
- **Confidence matched:** the global gate retains which candidate it selects,
  but its absolute logit is set to the local gate's mean absolute logit.

The latter is a diagnostic intervention, not a separately trained model. It
isolates clue correctness from response amplitude and softmax saturation.

## Performance and raw scores

Values after `+/-` are 95% confidence half-widths across paired seeds.

| Calibration | Scope | Held-out MSE | Structural-choice accuracy | Confidence | Raw semantic | Raw structural |
|---|---|---:|---:|---:|---:|---:|
| Free | Local RRWP | 0.3748 +/- 0.0080 | 0.7493 +/- 0.0041 | 0.7357 +/- 0.0058 | 0.5644 +/- 0.0028 | 0.2251 +/- 0.0031 |
| Free | Global RRWP | 0.0894 +/- 0.0022 | 0.9174 +/- 0.0022 | 0.8499 +/- 0.0024 | 0.5624 +/- 0.0036 | 0.1545 +/- 0.0018 |
| Matched | Local RRWP | 0.3748 +/- 0.0080 | 0.7493 +/- 0.0041 | 0.7357 +/- 0.0058 | 0.5644 +/- 0.0028 | 0.2251 +/- 0.0031 |
| Matched | Global RRWP | 0.2163 +/- 0.0080 | 0.9174 +/- 0.0022 | 0.7357 +/- 0.0058 | 0.5637 +/- 0.0034 | 0.2251 +/- 0.0031 |

The freely fitted global model reduces MSE by 76% relative to the local model.
After confidence matching it still reduces MSE by 42%. The remaining advantage
comes from selecting the correct candidate more often, not from applying a
larger gate.

Raw semantic scores already align in the free comparison. Raw structural
scores do not: the more confident global gate has the *smaller* projected score.
This is the expected clean-Jacobian saturation effect. Once confidence is
matched, both semantic and structural raw scores align while a large
performance gap remains.

The paired local-minus-global differences make the scale explicit. In the free
comparison they are `0.00204 +/- 0.00187` for semantic score and
`0.07061 +/- 0.00165` for structural score. After matching they are
`0.00071 +/- 0.00121` and `0.00000 +/- 0.00000`, respectively. These are
descriptive paired intervals; no post-hoc equivalence threshold is imposed.

## Distance-dependent score breakdown

Both models use the same fixed two-step lazy random-walk carrier map. This gives
genuine score mass at distances zero, one, and two while holding response
propagation constant. The profiles below are normalized to sum to one; their raw
masses reconstruct the corresponding total scores exactly.

| Channel and calibration | Local profile at d=(0,1,2) | Global profile at d=(0,1,2) | Total variation | Largest 95% CI half-width |
|---|---|---|---:|---:|
| Semantic, free | (0.58775, 0.30000, 0.11225) | (0.58807, 0.30000, 0.11193) | 0.00032 | 0.00011 |
| Semantic, matched | (0.58775, 0.30000, 0.11225) | (0.58908, 0.30000, 0.11092) | 0.00133 | 0.00011 |
| Structural, free | (0.59167, 0.30000, 0.10833) | (0.59167, 0.30000, 0.10833) | < 1e-12 | < 1e-12 |
| Structural, matched | (0.59167, 0.30000, 0.10833) | (0.59167, 0.30000, 0.10833) | < 1e-12 | < 1e-12 |

There is no scientifically meaningful difference in the distance-dependent
point estimates or in their error bounds. The slightly narrower global semantic
interval is only a tiny numerical difference. The structural profile is exact
up to floating-point precision because the registered kernel and normalized
source weighting are fixed by construction.

This null is expected and informative. RRWP scope changes the quality of the
structural clue available at a candidate; it does not change where the
post-encoding semantic response is carried. Distance measures the latter.

### Post-run stress test

A smaller four-seed check repeated the fit at local-clue reliabilities 0.65,
0.75, and 0.85 (4,096 train/test examples and 3,000 steps). The
confidence-matched local/global MSE pairs were `0.451/0.237`, `0.371/0.214`,
and `0.248/0.121`. Across all three settings the largest local-global profile
total variation was 0.00181 and the largest profile 95% CI half-width was
0.00012. The conclusion is therefore not specific to the registered 0.75
reliability point.

## Scientific implication

The experiment separates three quantities that can otherwise be conflated:

1. **Predictive information:** global RRWP selects the correct candidate more
   often and lowers held-out error.
2. **Projected-response scale:** raw structural score changes with confidence
   and saturation, even in the opposite direction to performance.
3. **Response geometry:** the distance profile remains unchanged when the two
   models share the same carrier mechanism.

Therefore, if the local/global ZINC models differ in performance but have
similar raw scores and distance profiles, the defensible conclusion is not that
RRWP scope is irrelevant. It is that the performance benefit is compatible
with better structural evidence being consumed by an otherwise similarly
organised response mechanism.

## Limits and next commitment

This task is graph-derived, but the fitted component is intentionally a linear
candidate-local gate rather than a full GRIT. The carrier map is fixed so that
the estimand can be identified cleanly. The figure is therefore a mechanistic
possibility result and an interpretation control, not evidence that trained
molecular models behave this way.

The next worthwhile experiment is now the smallest paired learned-model test:
train local-message GRIT models with local versus global RRWP on the same split,
first establish a held-out performance gap, then measure raw semantic and
structural totals and hierarchical distance profiles on paired graphs. Report
confidence/logit scale alongside the raw scores. The synthetic predicts that
performance can separate even if confidence-matched raw scores and normalized
distance profiles do not.
