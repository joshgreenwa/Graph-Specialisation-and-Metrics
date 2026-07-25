# Canonical implementation

The normative scientific specification is
[`../README.md`](../README.md), protocol
`donor-swap-specialisation-carriage-v3`. This package is its public implementation. Historical
implementations remain provenance rather than public alternatives. The selected internal
`carriage/` and `specialisation/` modules supply only checkpoint-compatible low-level GRIT
machinery where explicitly imported.

## Public entry points

Local or installed use:

```python
from graph_specialisation_metrics import MethodologyConfig, main

main(MethodologyConfig(
    tasks=("zinc", "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode"),
    train_seeds=(42,),
    checkpoints={
        # Optional explicit paths; otherwise each registered Drive results directory is searched.
        # "zinc:42": "/path/to/best.ckpt",
    },
))
```

Colab use:

```python
from graph_specialisation_metrics.methodology.colab import run

run(
    tasks=("zinc", "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode"),
    train_seeds=(42,),
    phases=("scores", "causal", "carriage", "figures"),
)
```

The paste-ready clone/mount/dispatch cell is
[`../../../experiments/methodology/canonical_methodology_colab.py`](../../../experiments/methodology/canonical_methodology_colab.py).

## Module boundary

| Module | Responsibility |
|---|---|
| `protocol.py` | Version, fixed constants, split discipline, fingerprints. |
| `tasks.py` | Output geometry, loss, content/PE/support declarations, GRIT task registration. |
| `sampling.py` | Exact graph-uniform/node-uniform semantic law and node-uniform structural law. |
| `interventions.py` | One-row semantic replacement and dense-equivalent sparse structural footprint copy. |
| `backend.py` | Native GRIT `wV`/final-state capture, z-space Jacobians, ablation and patching. |
| `scores.py` / `distance.py` | Raw event score, hierarchy, coordinates, and exact SPD accounting. |
| `carriage.py` | `F_sens` and positive-is-beneficial donor-wise integrated `B`. |
| `causal.py` / `validation.py` | Clean necessity, donor-wise necessity, gross patch responses, rescue/induction, controls. |
| `bootstrap.py` | Fixed 2,000-draw nested percentile intervals and 20% graph-trimmed summaries. |
| `cache.py` | Atomic, contract-bound Drive caches that reject stale methodology/results. |
| `audit.py` | Soft numerical/estimability audits: record, log, and continue; strict mode raises. |
| `figures.py` | Shared publication theme, exact labels, intervals, PDF+PNG+metadata export. |
| `runner.py` | Stage orchestration for dense, 1-hop, k-hop, and k-hop+VNode registrations. |

Task-specific presentation can be added without changing cached measurements:

```python
from graph_specialisation_metrics.methodology.figures import (
    register_task_figure_modifier,
)

def adjust_zinc(name, figure, axes):
    if name == "structural_vs_semantic_scores":
        axes.set_title("ZINC")

register_task_figure_modifier("zinc", adjust_zinc)
```

Scientific task differences are registrations, not runner forks. A task must declare all semantic,
structural, fixed-support, output-scaling, carrier, loss, split, and tolerance boundaries listed
in Section 12 of the normative README.

## Output layout

```text
<output>/
  protocol.json
  index.json
  audits.json                    # every soft audit failure, by task/seed
  <task>/
    seed_<training-seed>/
      model.json
      audits.json                # this run's soft audit failures
      audit/
      cache/{scores,carriage,causal}/
      figures/
        *.pdf
        *.png
        *.metadata.json
      figures.json
    population.json              # seed estimates; population CI only with >=3 seeds
```

Numerical, invariance, and estimability audits report and continue by default: a breach is logged
once, merged by audit name with its worst observed value and repeat count, and written to the two
`audits.json` files (model checks also land in `canonical_audits.failures` of `model.json`).
`run(..., strict_audits=True)` restores fail-closed runs, raising `AuditError` on the first breach.
Cache fingerprints are unaffected by this switch, so strict and reporting runs share caches.

Training checkpoints and dataset caches are read-only. Analysis caches bind the protocol
fingerprint, checkpoint SHA-256, task adapter, output representation and sigma, model geometry,
split IDs, event manifest, donor/source dose, bootstrap seed, `F_sens`, and the positive-beneficial
sign convention.

The score cache also retains clean attention distance mass and frozen-family exact/support-
normalized score profiles as descriptive diagnostics. These never replace the transport score.
When at least three training seeds are run, `population.json` bootstraps seed-level summary and
association estimates without aligning head indices; with fewer seeds it records the individual
seed estimates and explicitly suppresses a seed-population interval.
