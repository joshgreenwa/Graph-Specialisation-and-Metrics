# Dense ZINC and QM9 GRIT causal head populations

## Scope

This analysis repeats the PCQM4Mv2 causal-population study for the registered dense
`zinc` and `qm9_gap_dense` GRIT checkpoints at training seed 42. It deliberately excludes the
1-hop, 2-hop, and virtual-node controls: those models remain separate registered tasks and are not
pooled with the dense checkpoints.

The discovery variables and causal estimands are unchanged:

1. semantic and structural specialists are selected using discovery-only `J` and `D_rel`;
2. semantic/structural heads are matched on `J`, with layer distance used only as a tie-break;
3. every selected specialist receives a distinct active, non-selected `J`-matched control;
4. restoration inserts the clean routed head output into an intervened molecule;
5. injection inserts the intervention-state routed head output into a clean molecule;
6. necessity independently ablates the head in the clean and intervened runs; and
7. the all-head clean-ablation sweep tests `J` against held-out output movement, both raw and with
   layer fixed effects.

GRIT interventions attach to its native routed per-head output (`wV`), before the model mixes
heads. This is the GRIT analogue of the Graphormer transport site used by the PCQM4Mv2 runner; it
does not reinterpret GRIT's relation/RRWP-conditioned attention as an additive structural bias.

## Frozen population and sampling contract

The dense-GRIT launcher requests 8 semantic/structural pairs and requires at least 6. This is the
registered architecture-adapted population gate for the 80-head models and is attainable for the
smaller ZINC discovery specialist pool. Candidate selection, Hungarian matching, null assignment,
and the `D_rel = +/-0.10` and `J >= 0.20` regions remain fixed before causal outcomes are loaded.
If a checkpoint cannot supply that population, the runner saves and reports a `not_estimable`
gate and continues to the other task.

Each task uses mutually disjoint populations of 256 discovery, 256 causal-event, and 256
clean-ablation molecules, plus 2,000 semantic donor molecules. Six sources and eight donors per
source produce 48 intervention events per molecule and channel. The registered nested bootstrap
uses 2,000 draws and 95% intervals. Results describe one trained checkpoint per task; they do not
estimate training-seed variation.

## Execution and caches

Use
[`../experiments/methodology/grit_dense_causal_population_colab.py`](../experiments/methodology/grit_dense_causal_population_colab.py).
The launcher reuses the task registry's ZINC/QM9 training checkpoint and dataset roots read-only,
while writing new artifacts below:

```text
grit_dense_causal_population_paper/
  zinc/seed_42/
  qm9_gap_dense/seed_42/
```

`PHASE = "run"` populates or resumes per-graph shards, `"all"` also renders figures, and
`"figures"` reads CPU caches without cloning GRIT, rebuilding RRWP, loading a checkpoint, or
executing model forwards. Each task has its own model record, score/gate/event/ablation caches,
figure directory, and `focused_causal_population_manifest.json`.

Resume checks consolidated scores and both discovery gates before constructing donor manifests.
An exact completed core therefore proceeds directly to figures. When a compatible completed run
is retargeted from the former 16/12 population policy to 8/6, the runner filters and imports the
already-measured per-head causal rows, retains the old cache contract as read-only lineage, reuses
the gate-independent all-head clean ablations, and builds donor manifests only for graph/head rows
that are genuinely absent.

The exported suite matches the PCQM4Mv2 population analysis: donor-averaged primary causal tests,
correct-pairing advantage, mismatch-adjusted robustness, `J` versus clean ablation, continuous
`D_rel` versus causal preference, and the discovery-only matching audit. Plot titles and provenance
identify Dense ZINC GRIT or Dense QM9 GRIT explicitly.
