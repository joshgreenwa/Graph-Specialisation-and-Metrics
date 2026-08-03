# Chapter 6 synthetic iteration: what specialisation can identify

## Decision

The metric-level synthetic supports the revised Chapter 6 hierarchy:

1. **Performance should remain primary.**
2. **Specialisation can describe channel organisation.**
3. **Canonical distance can describe response propagation.**
4. Neither specialisation nor response distance measures the upstream topology
   on which a precomputed structural encoding depends.

The next scientific investment should therefore be the support-by-RRWP-horizon
factorial, not a larger catalogue of specialisation profiles.

## Experiment 1: matched mechanisms

Every example has two independent semantic values and a structural clue that
selects one candidate. Three mechanisms are compared across eight paired seeds:

| Mechanism | Clue | Semantic carrier | PE dependency origin |
|---|---|---:|---:|
| Local + partial short PE | correct with probability 0.75 | distance 0 | distance 1 |
| Local + exact multi-hop PE | always correct | distance 0 | distance 4 |
| Exact PE + semantic transport | always correct | distance 4 | distance 4 |

The exact multi-hop and semantic-transport mechanisms are behavioural twins.
Only the registered carrier location differs, making the latter a positive
control for the distance estimator.

The gate parameter is the exact empirical MSE optimum under a fixed confidence
ceiling. Candidate values, targets, and test examples are paired across
conditions.

### Results

| Quantity | Partial short PE | Exact multi-hop PE | Semantic transport |
|---|---:|---:|---:|
| Held-out MSE | 0.3751 +/- 0.0069 | 0.000050 +/- 0.000001 | 0.000050 +/- 0.000001 |
| Raw semantic score | 0.5635 +/- 0.0034 | 0.5631 +/- 0.0035 | 0.5631 +/- 0.0035 |
| Raw structural score | 0.2348 +/- 0.0023 | 0.0297 +/- 0.0002 | 0.0297 +/- 0.0002 |
| Structural-swap loss increase | 0.1263 +/- 0.0063 | 0.5014 +/- 0.0067 | 0.5014 +/- 0.0067 |
| Semantic response distance | 0 | 0 | 4 |
| Structural response distance | 0 | 0 | 0 |

Values after `+/-` are 95% confidence half-widths across paired seeds.

## Do specialisation metrics differ?

### Normalized head coordinates: no

For pure semantic and structural heads, all three mechanisms give:

- semantic head: `D_rel = +1`, `J = 1`;
- structural head: `D_rel = -1`, `J = 1`.

The result is not an artifact of using only pure heads. A four-head robustness
layout with semantic/structural loadings `(1,0)`, `(0.75,0.25)`, `(0.25,0.75)`,
and `(0,1)` gives `D_rel = (+1,+0.5,-0.5,-1)` and `J = 1` for every mechanism
and every clue reliability.

This invariance is expected. Channel-wise normalization removes global response
amplitude, while `D_rel` records the relative semantic-versus-structural division
of labour. It does not encode whether the structural variable is informative
about the target.

### Raw scores: yes, but in the wrong direction for a usefulness claim

The raw semantic score is essentially unchanged. The raw structural score is
about eight times **smaller** for the exact clue, even though its performance and
structural-swap loss effect are much better.

The exact gate is confident and locally saturated. Canonical head scores project
the finite state difference through the clean Jacobian, so the structural
gradient becomes small near a confident softmax endpoint. A finite structural
swap still produces a large output and loss change. Raw projected response is
therefore not a monotonic scale of task usefulness.

## Do the semantic/structural distance breakdowns differ?

### Short versus multi-hop local preprocessing: no

Both interventions replace payload at a candidate and both mechanisms consume
that payload at the candidate. Their semantic and structural score mass is
therefore at source-to-carrier distance zero.

This is correct under the canonical estimand. Its distance origin is the node
whose already-computed payload is replaced. It does not recursively trace which
edges or nodes influenced that payload during preprocessing. Multi-hop RRWP may
depend on topology four or more hops away while structural response distance
remains zero.

### True semantic transport: yes

The behavioural transport twin moves semantic score mass from distance zero to
distance four while leaving structural mass at zero. This positive control shows
that the distance null is not caused by an insensitive implementation. The
metric detects response propagation when response propagation actually changes.

## Experiment 2: clue-quality sweep

Clue reliability was swept through `0.55, 0.65, 0.75, 0.85, 0.95, 1.0`.

- Test MSE decreases monotonically from 0.496 to 0.00005.
- The loss increase caused by a structural swap rises from 0.005 to 0.501.
- Raw semantic score remains near 0.563.
- Raw structural score is non-monotonic: it rises from 0.056 to 0.249 at
  reliability 0.85, then falls to 0.030 for the exact clue.
- All four normalized `D_rel`/`J` head coordinates remain unchanged.

The sweep separates three questions that should not be conflated:

1. **What channel does a head respond to?** `D_rel` and `J`.
2. **How much local projected response is present?** Raw specialisation score.
3. **Is the response aligned with the task?** Held-out loss and label-aware
   finite interventions.

## Implication for the existing ZINC null

Similar short-RRWP and multi-hop-RRWP specialisation or carriage profiles are
not evidence that multi-hop RRWP is unused. They are compatible with:

- the same head-level division of labour;
- the same candidate-local semantic response geometry; and
- very different structural clue quality and predictive performance.

The null says that the fitted models have similar **response organisation under
the registered interventions**. It does not say that their preprocessing
horizons, available structural information, or task-aligned use are equivalent.

Conversely, a raw-score difference would not by itself explain the performance
gap because raw scores are scale- and saturation-sensitive.

## Recommended measurement package

For the Chapter 6 factorial, report:

- held-out performance and equivalence tests as the primary evidence;
- an explicit preprocessing-dependency audit, such as remote-edge RRWP
  sensitivity or RRWP horizon/diameter coverage;
- at most one semantic response-distance figure, interpreted as propagation;
- normalized specialisation only as a descriptive organisation result; and
- finite task-facing structural ablations only if their counterfactual meaning
  is defensible.

Do not reinterpret canonical structural score distance as RRWP dependency
horizon. Those are different estimands.

## Next experiment before the full ZINC run

The remaining useful lightweight bridge is a tiny learned graph-native
support-by-horizon factorial with actual attention heads:

- local messages + short RRWP;
- local messages + multi-hop RRWP;
- dense messages + short local-pair RRWP and neutral nonlocal pairs; and
- dense messages + multi-hop local-pair RRWP and neutral nonlocal pairs.

It should reuse the cycle/chain task, capture real layer/head states, and apply
the same score code. Its purpose is implementation validation: verify that the
metric null survives a learned multihead model and that dense-neutral support
does not accidentally receive distant pair coordinates. It should not be
expanded if the clean ZINC arms are already ready to run.
