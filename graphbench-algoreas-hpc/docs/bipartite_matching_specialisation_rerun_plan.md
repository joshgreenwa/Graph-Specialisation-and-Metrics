# Bipartite matching GRIT structural-PE refinement

The focused causal-validation specification for the locked method is maintained
in
[`bipartite_matching_causal_validation_plan.md`](bipartite_matching_causal_validation_plan.md).

**Status:** refinement complete; production choice locked
**Last updated:** 2026-07-30
**Implementation branch:** `expansion/graphormer_specialisation`  
**Protocol:** `graphbench-bipartite-pe-refinement-v1`  
**Task:** `graphbench_bipartite_matching_hard` only  
**Models:** official-GRIT checkpoints, training seeds `0,1,2,3`  
**Analysis population:** GraphBench `n=16` validation split  

## 1. Decision this run must make

The first matching analysis found strong causal validation of Joint sensitivity `J`, a promising
Semantic score, and weaker/noisier Structural-score validation. Most heads lie close to the
Semantic/Structural diagonal, with a small semantic-specialist tail and little structural tail.
The aim of this run is to select one defensible structural intervention and one output-projected
score system before returning to carriage or distance analyses.

The semantic reciprocal-edge-unit donor swap is fixed. Topology, edge values, node types, labels,
and output/readout coordinates remain fixed during every structural intervention.

**Locked production choice:** `complete_pe_copy` with `coherent` output-movement scoring.
Node transposition is rejected. RRWP-only donor-copy remains a positive ablation and transport mass
remains a secondary diagnostic.

## 2. Registered four-arm factorial

Every arm receives exactly the same ordered source/donor node pairs.

| Arm | RRWP operation | Explicit degree operation |
|---|---|---|
| `rrwp_copy` | donor row/column/self copied onto source | unchanged |
| `rrwp_transpose` | source/donor permutation applied to both RRWP axes | unchanged |
| `complete_pe_copy` | donor row/column/self copied onto source | donor degree copied onto source |
| `complete_pe_transpose` | source/donor permutation applied to both RRWP axes | source/donor degrees transposed |

`log_deg` is not independently edited: the official GRIT adapter derives it exactly from the
intervened degree. For these checkpoints, complete model-visible structural input is the first 16
RRWP channels (both node diagonal and full pair tensor), degree, and log degree. The cache's
17th RRWP channel is not consumed by official GRIT and remains unchanged. SPD/RWSE are not
consumed by this GRIT path. Bond/edge values are semantic inputs and stay fixed.

## 3. Common source/donor law

- Enumerate every eligible node source.
- Require donor and source to have the same node type and inferred bipartition side.
- Exclude the source itself and identical RRWP-role footprints.
- Do not degree match.
- Partition candidates into near/middle/far clean RRWP-role-distance strata.
- Draw across strata without replacement, up to eight donors.
- Never duplicate donors to fill a ragged source.
- Freeze manifests without the training seed in the RNG key so all four checkpoints receive the
  same graphs, sources, donors, and Taylor subset.

The paired manifest retains degree gap. Calibration is repeated for gap `0`, gap `1`, and gap
`2+`, so the former degree-matched regime is available as a prespecified sensitivity slice without
shrinking the donor pool or changing pairs between arms.

Cache graph/source/donor IDs, inferred side, node type, degrees, degree gap, footprint hashes,
RRWP-role distance and stratum, eligible-pool size, realised count, exhaustion, RRWP RMS dose,
degree/log-degree dose, and a standardised full-input dose.

## 4. Two registered score systems

For projected event/head/carrier vectors

```text
q[e,l,h,i,t] =
  <d z_t / d o[l,h,i] at clean, o_clean[l,h,i] - o_event[l,h,i]>
```

compute:

```text
M[e,l,h] = sum_i ||q[e,l,h,i,:]||_2
C[e,l,h] = ||sum_i q[e,l,h,i,:]||_2
R_coherence[e,l,h] = C[e,l,h] / M[e,l,h]
R_cancel = 1 - R_coherence
```

`M` is transport mass. `C` is coherent first-order output movement. The ratio is computed at event
level before donor/source/graph aggregation; events below the registered mass floor are
non-estimable rather than zero.

Derive independent coordinate systems:

```text
J_M, D_rel_M from semantic/structural M
J_C, D_rel_C from semantic/structural C
```

Never mix score systems inside one coordinate.

## 5. Causal validation

For all 48 heads, retain:

- matched bidirectional restoration/injection gross response;
- same-source, alternative-donor mismatch response matched by full input dose;
- mismatch-adjusted response `G = P_matched - P_mismatch`;
- aligned and gross donor-wise necessity;
- clean single-head ablation prediction movement and loss change; and
- exact injection vectors on the Taylor subset.

Primary validations, within each seed:

```text
S_sem -> G_sem
S_str -> P_str,matched
S_str -> P_str,mismatch
S_str -> G_str
J -> 0.5 * (G_sem/a_sem + G_str/a_str)
D_rel -> G_sem/a_sem - G_str/a_str
J -> clean ablation and donor-wise total necessity
D_rel -> donor-wise necessity contrast
```

The four-way channel matrix reports all of `S_sem -> G_sem`, `S_sem -> G_str`,
`S_str -> G_sem`, and `S_str -> G_str`. The off-diagonal cells are the direct specificity
controls; the intended result is stronger on-channel than cross-channel calibration, not merely a
large response to either intervention.

`D_rel` is interpreted only for heads with `J >= 0.20`. Report its SD/IQR and balanced/semantic/
structural-tail counts so range restriction is visible. Do not manufacture binary specialisation
when the learned population is genuinely balanced.

## 6. Taylor-fidelity audit

Freeze 12 refinement graphs. For semantic events and every structural arm:

```text
m_pred = -sum_i q[e,l,h,i,:]
m_exact = z_clean+event(l,h) - z_clean
```

Cache cosine fidelity, norm ratio, relative error, predicted/exact norms, dose, degree gap,
channel, arm, layer, head, and estimability. Taylor thresholds are soft audits.

## 7. Replication and lockbox

Run all four trained seeds. Never pool 192 heads as independent observations and never average
corresponding head identities across seeds.

- Compute head correlations and within-layer permutation tests separately per seed.
- Combine the four seed correlations using Fisher-z means.
- Report every seed, seed-level t intervals, sign consistency, ranges, and leave-one-seed-out
  estimates.
- Use seed-stratified, within-layer permutations for the aggregate association.

The 48 causal graphs are split before outcomes are inspected:

- `refinement`: 24 graphs, all eight arm/score candidates visible;
- `confirmation`: 24 graphs, blindly cached but inaccessible to the finalizer until a selection
  lock names one arm and one score system.

The refinement finalizer ranks candidates using the registered hierarchy:

1. integrity, causal-control support, and Taylor support;
2. replicated `S_str -> G_str`;
3. replicated `D_rel ->` causal channel contrast;
4. preservation of `J ->` total causal response and clean necessity;
5. Taylor fidelity;
6. carrier coherence as a tie-break.

The rank is advisory and does not write the lock automatically.

## 8. Production sizes

| Component | Size per seed |
|---|---:|
| discovery graphs | 64 |
| refinement causal graphs | 24 |
| confirmation causal graphs | 24 |
| clean-ablation graphs | 64 |
| semantic sources | up to 12 reciprocal edge units/graph |
| structural sources | all eligible nodes |
| donors | up to 8 unique donors/source |
| Taylor subset | 12 refinement graphs |
| permutation replicates | 2,000 |

## 9. Runtime, caching, and jobs

Output root:

```text
/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/
  grit_specialisation_bipartite_pe_refinement_v1/
```

Logical layout:

```text
graphbench_bipartite_matching_hard/
  seed_0/
    common/
      clean_jacobians/{discovery,taylor}/
      scores/semantic/
      causal/{refinement,confirmation}/semantic/
      clean_ablation/
    arms/
      rrwp_copy/
      rrwp_transpose/
      complete_pe_copy/
      complete_pe_transpose/
    progress/
    workers/
  seed_1/
  seed_2/
  seed_3/
  analysis/refinement/
  selection_lock.json
  analysis/confirmation/
```

All graph shards are atomic and protected by protocol, checkpoint, split, repository, namespace,
and event-manifest contracts. Figures can be regenerated without GPU work. Common discovery and
Taylor Jacobians are computed once per seed and reused by all four arm jobs.

Production is a three-stage Slurm DAG:

1. one 12-element common array: three components by four seeds;
2. one 16-element arm array: four arms by four seeds, after all common elements succeed;
3. one model-free CPU refinement finalizer.

Every GPU element requests one untyped Ampere GPU and at most six hours. The production default
starts at 16 graph event groups per score forward and 24 independently patched heads per causal
forward (two saturated passes over 48 heads), with automatic OOM backoff. Heartbeats report
progress and GPU/VRAM telemetry. These execution-only batch sizes do not alter the scientific
fingerprint or invalidate completed graph shards.
The vectorized finalizer requests one CPU for six hours (six CPU-hours maximum), keeping it within
the remaining CPU allocation without changing any estimator.

## 10. Acceptance policy

Hard failures:

- missing/wrong checkpoint or official GRIT commit;
- wrong task/split/model geometry;
- cache-contract or event-manifest mismatch;
- intervention changing undeclared fields;
- tensor/patch geometry mismatch;
- non-finite values entering a registered estimator; or
- missing required shards after component completion.

Soft audits:

- declared no-op tolerance;
- same-condition patch tolerance;
- low score/effect/Taylor support;
- low Taylor fidelity;
- unavailable mismatch controls;
- ragged donor support; and
- reporting-floor warnings.

Soft audits are cached and flagged but do not terminate production.

## 11. Implementation checklist

- [x] Four structural interventions.
- [x] Analysis-only degree override with unchanged training behaviour.
- [x] Common paired source/donor manifests with role and dose metadata.
- [x] Transport-mass and coherent score systems.
- [x] Event-level carrier coherence/cancellation.
- [x] Replica-specific all-head batched patching.
- [x] Matched, mismatch, adjusted, necessity, and clean-ablation endpoints.
- [x] Taylor prediction/exact-injection audit.
- [x] Four-seed nested inference and range-restriction reporting.
- [x] Refinement/confirmation lockbox.
- [x] Atomic common/arm caches and component progress.
- [x] Slurm common/arm/finalizer DAG with six-hour GPU and CPU safety limits.
- [x] Complete local unit/static/contract verification (`198 passed, 1 skipped`).
- [x] Refresh HPC checkout and run new preflight.
- [x] Queue production DAG.
- [x] Inspect refinement outputs and write selection lock.
- [ ] Run confirmation finalizer for the locked candidate.

## 12. Results log

| Date | Version | Component | Status | Observation |
|---|---|---|---|---|
| 2026-07-27 | draft | earlier degree-law rerun | superseded | Replaced by four-arm PE factorial |
| 2026-07-28 | v1 | implementation | ready | Dedicated matching-only core runner; full local suite passes |
| 2026-07-30 | v1 | refinement decision | locked | Complete-PE donor-copy + coherent score selected; transposition rejected |
