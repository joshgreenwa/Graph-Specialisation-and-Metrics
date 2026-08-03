# Chapter 6 learned-head synthetic: RRWP-guided semantic relay

## Post-hoc audit (supersedes the original alignment interpretation)

The performance contrast below remains valid, but the original claim that
global RRWP improves within-head semantic--structural distance alignment is not
secure. A later audit found that a same-layer other-head null explains most of
the raw cosine, while the global structural intervention has about 2.7 times
the input-space norm of the local intervention. Dividing the finite response by
event norm reverses the local/global alignment ordering. The fixed relay route
and per-node auxiliary loss also constrain carriage geometry by construction.

Accordingly, the raw matched-head cosine and close carriage profiles below
should be treated as descriptive properties of this task, not evidence for a
general mediation-quality signature. The independent molecular-sites follow-up
uses an ordinary all-node readout, explicit event-dose reporting, a same-layer
head null, and direct site-selection fidelity. Its report is
`chapter6_rrwp_molecular_sites_pilot.md`.

## Decision

This is the first local synthetic in the Chapter 6 sequence that genuinely
measures learned per-head semantic and structural score distance. Five-seed
runs at two local-cue reliabilities give a clear, controlled performance
contrast and reproduce only part of the empirical ZINC pattern:

- semantic carriage geometry is similar across local and global RRWP;
- learned heads are overwhelmingly generalists;
- raw semantic and especially structural scores are larger under global RRWP;
- structural carriage is substantially larger under global RRWP, although its
  normalized route geometry becomes almost identical when the local cue is
  stronger;
- global RRWP more consistently aligns semantic and structural distance
  profiles within the learned heads; and
- local RRWP does **not** have wider scale-normalized score intervals.

The last result is useful. Wider local-RRWP intervals are not an automatic
consequence of lower-quality PE. They require a more specific explanation in
the molecular models.

## Task and fairness controls

Each graph has two scalar-valued candidates connected to a selector, relay, and
root. One candidate lies at the base of a remotely closed cycle and the other
at the base of two open arms. The target is the value on the cycle candidate.

Candidate degree supplies an imperfect local cue. The primary run makes it
correct on 75% of examples and a robustness run raises this to 90%; both use
weak and strong margins. The cycle closure is four hops from its candidate,
beyond the three learned message layers, but appears in higher-order RRWP.
Thus ordinary message passing cannot reach the closure within the model depth.

All arms use the same 15,013-parameter, three-layer, four-head, 1-hop attention
model. Every RRWP projection has 13 input channels:

| Arm | Visible RRWP | Scientific role |
|---|---|---|
| Local | `I,P`; channels 2--12 zeroed | noisy local structural cue |
| Global | `I,...,P^12` | local messages with remote structural context |
| Shuffled global | local channels intact; higher channels independently swapped | controls feature count and scale without useful remote information |

RRWP channels are scaled once from the registered training-support
distribution. The same scaling and parameterization are used in every arm.
Five seeds use 2,048 training, 512 validation, and 1,024 held-out examples.

## 1. Performance differs under a fair control

| Arm | Held-out MSE | 95% seed CI half-width |
|---|---:|---:|
| Local RRWP | 0.3987 | 0.0061 |
| Global RRWP | 0.0112 | 0.0121 |
| Shuffled global RRWP | 0.3989 | 0.0042 |

The useful higher-order information, rather than additional nonzero channels or
parameters, produces the performance gain. Shuffling those channels returns
performance to the local baseline. Four global seeds reach MSE 0.0046--0.0056;
one reaches 0.036, so optimization stability is a real part of the result and
is exposed rather than filtered.

## 2. Genuine per-head raw scores

For every learned routed head output `z[layer,node,head,width]`, semantic and
structural donor events use the canonical projection

`q = (z_clean - z_event) * dy/dz_clean`.

Mass is binned using the exact shortest-path distance from the changed
candidate to each carrier. Distance contributions reconstruct each raw head
score exactly.

| Arm | Mean raw semantic score | Mean raw structural score | Head-score correlation |
|---|---:|---:|---:|
| Local RRWP | 0.0951 | 0.00544 | 0.665 |
| Global RRWP | 0.1210 | 0.0626 | 0.834 |

The raw distributions do not align. Global semantic score is about 1.27 times
larger and global structural score about 11.5 times larger. The global structural
PE event has 2.74 times the input-space norm of the local event because it
contains more active RRWP channels, so event dose explains part but not all of
the difference.

After within-model channel normalization, 88.3% of local heads and 90.0% of
global heads fall inside the preregistered generalist region
`|D_rel| <= 0.5`. Only 5.0%/1.7% are structural-leaning and 6.7%/8.3% are
semantic-leaning (local/global). This reproduces the distributed-generalist
observation without hard-coding head roles.

Semantic and structural distance vectors are positively aligned but do not
always share an exact peak. Across the actual seed-specific learned heads,
mean profile cosine is 0.678 locally and 0.725 globally; peak distance agrees
for 16.7% and 48.3% of heads, respectively. Global cosine is higher in four of
five paired seeds and peak agreement is higher in four seeds and tied in one.
This is promising architecture-level evidence for the proposed **quality of
structural mediation** mechanism: global RRWP aligns structural and semantic
transport more reliably, not merely more strongly. It is not yet a validated
quality metric: cosine alignment does not monotonically predict MSE across the
five global seeds, although the failed reliability-0.90 seed has the lowest
global peak-match fraction.

## 3. Does Functional carriage match?

Final-state carriage is computed separately from head scores. Because the route
readout is linear, the finite carrier contributions reconstruct output movement
exactly and equal the straight-line Functional carriage integral.

| Channel | Local total | Global total | Normalized-profile TV |
|---|---:|---:|---:|
| Semantic | 0.533 | 0.689 | 0.042 |
| Structural | 0.0201 | 0.390 | 0.103 |

Semantic carriage follows nearly the same route: both models place most mass at
distances one and two and decline at distance three. Its total scale is larger
globally, but normalized geometry is close.

Structural carriage does not match in total scale: global RRWP carries about
19 times as much structural effect. Its normalized route geometry is only
moderately different in the primary run and becomes nearly identical at 90%
local-cue reliability. Thus carriage *shape* can match while the amount of
structural mediation remains very different.

## 4. Are local score estimates less confident?

No. Raw interval width is larger globally because global scores are larger. To
remove this scale effect, every graph/layer/head profile is normalized across
distance before the nested seed/graph bootstrap.

| Channel | Local mean 95% profile-interval width | Global mean | Local/global ratio |
|---|---:|---:|---:|
| Semantic | 0.0838 | 0.1354 | 0.619 |
| Structural | 0.1668 | 0.1983 | 0.841 |

Both semantic and structural profile intervals are narrower locally. The
molecular observation of wider local-RRWP intervals is therefore not a generic
signature of poorer structural information. Plausible remaining causes include
between-seed solution instability, graph-support heterogeneity, weak-cell
normalization, or a task-specific mixture of routes.

## 5. Reliability iteration

Only the probability that the local degree cue is correct changes from 0.75 to
0.90. Architecture, parameters, graph topology, sample sizes, and optimization
remain fixed.

| Quantity | Reliability 0.75 | Reliability 0.90 |
|---|---:|---:|
| Local / global / shuffled MSE | 0.399 / 0.011 / 0.399 | 0.210 / 0.049 / 0.211 |
| Semantic carriage profile TV | 0.042 | 0.034 |
| Structural carriage profile TV | 0.103 | 0.014 |
| Semantic profile-width ratio, local/global | 0.619 | 0.869 |
| Structural profile-width ratio, local/global | 0.841 | 0.745 |
| Local / global semantic-structural cosine | 0.678 / 0.725 | 0.678 / 0.752 |
| Local / global peak-match fraction | 0.167 / 0.483 | 0.333 / 0.500 |

The stronger local cue improves local and shuffled performance as expected, but
the global arm remains better. One of five global seeds at reliability 0.90
fails to learn the remote cue (MSE 0.210); the other four score 0.004--0.014,
giving a median global MSE of 0.013. This failure makes the mean conservative
and warns against a three-seed conclusion.

The important qualitative results survive: raw structural scores remain much
larger globally, local score intervals remain no wider, and global
structural-semantic distance alignment remains higher. Meanwhile normalized
structural carriage converges almost exactly. This separates carriage geometry
from mediation amount and alignment quality.

## Scientific conclusion

The synthetic validates the central mechanism in a learned model: higher-order
RRWP can improve the structural selection of semantic values under strictly
1-hop learned messages. The later dose/null audit does not support treating
stronger raw semantic--structural profile alignment as the corresponding
general signature.

> Similar carriage geometry can coexist with a large RRWP performance benefit,
> but neither close carriage nor raw matched-head alignment identifies the
> performance mechanism on its own.

This is promising enough to carry forward, but the claim should be narrow.
Wider local intervals in ZINC should be treated as an empirical
model-population property, not as evidence of lower-quality mediation by
itself. The next experiment must use an independent task and a behavioural
measure of whether structure guides the correct semantic selection.

## Reproduce and inspect

Primary run:

```bash
PYTHONPATH=src python -m graph_specialisation_metrics.synthetic.rrwp_semantic_relay \
  --output-dir outputs/chapter6_rrwp_semantic_relay_s5_v2 \
  --seeds 0,1,2,3,4 \
  --local-clue-reliability 0.75
```

For the robustness run, change the output directory to
`chapter6_rrwp_semantic_relay_r90_s5_v2` and reliability to `0.90`. Every raw
measurement is saved as CSV. `--reanalyze-only` regenerates summaries and
figures without training. The cross-run figures and paired-seed alignment table
are in `outputs/chapter6_rrwp_semantic_relay_sensitivity_v1`.
