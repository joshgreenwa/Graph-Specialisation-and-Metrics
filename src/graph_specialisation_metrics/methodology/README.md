# Canonical implementation

The normative scientific specification is
[`../README.md`](../README.md), protocol
`donor-swap-specialisation-carriage-v4`. This package is its public implementation. Historical
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

Official PCQM4Mv2 Graphormer use:

```python
main(MethodologyConfig(
    tasks=("graphormer_pcqm4mv2",),
    train_seeds=(0,),
    checkpoints={},  # downloads the registered public checkpoint
    task_overrides={
        "graphormer_pcqm4mv2": {
            "dataset_root": "/path/to/pcqm4mv2",
            "cache_dir": "/path/to/huggingface-cache",
        },
    },
))
```

Official-GRIT GraphBench AlgoReas use is provided by the repository frontend
`graphbench-algoreas-hpc/bin/grit_specialisation.py`. It reuses the training runner's exact
GraphBench conversion, PE cache, upstream GRIT layers, prediction heads, and checkpoint geometry.
Because matching and flow expose weighted edges rather than swappable node-content rows, this is
recorded as protocol extension `graphbench-edge-semantic-v1`: semantic sources are edges and
structural sources remain nodes. Their bootstrap is paired at the graph level while independently
resampling the two source domains. Existing registered tasks keep the unmodified node-content v3
intervention and source-paired hierarchy.

`0` is a stable cache label for the one public model, not a claim about its training seed. A local
Hugging Face model directory can be supplied in `checkpoints`; a raw `.pt`, `.bin`, or
`.safetensors` state can also be overlaid when `model_id` resolves to the matching base
configuration. The checkpoint-compatible `transformers==4.40.2` stack supports Python 3.10-3.12;
the Colab installer rejects Python 3.13 with a direct compatibility message.

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
| `tasks.py` | Backend-neutral task contract and GRIT/Graphormer task registrations. |
| `sampling.py` | Exact graph-uniform/node-uniform semantic law and node-uniform structural law. |
| `interventions.py` | One-row semantic replacement and dense-equivalent sparse structural footprint copy. |
| `backend.py` | Native GRIT `wV`/final-state adapter. |
| `graphormer.py` | Official checkpoint/dataset loading, native Graphormer transport hooks, graph-token readout adapter. |
| `grit_figure_data.py` | Read-only canonical score/model validation plus contract-cached GRIT attention, selected/all-head routed-output PCA, raw-logit, and displayed-graph coordinate diagnostics. |
| `grit_figure_plots.py` | PCQM-aligned ZINC/QM9 presentation layer with RDKit attention rendering, layer-PCA grids, and publication exports. |
| `graphbench.py` | Exact GraphBench runner/checkpoint adapter, edge-semantic donor law, official-GRIT hooks, and nonlinear readout replay. |
| `scores.py` / `distance.py` | Raw event score, hierarchy, coordinates, and exact SPD accounting. |
| `carriage.py` | `F_sens` and positive-is-beneficial donor-wise integrated `B`. |
| `causal.py` / `validation.py` | Clean necessity, donor-wise necessity, gross patch responses, rescue/induction, controls. |
| `bootstrap.py` | Fixed 2,000-draw nested percentile intervals and 20% graph-trimmed summaries. |
| `cache.py` | Atomic, contract-bound Drive caches that reject stale methodology/results. |
| `audit.py` | Soft numerical/estimability audits: record, log, and continue; strict mode raises. |
| `execution.py` | Runtime-only multi-graph batching and CUDA-OOM backoff. |
| `figures.py` | Shared publication theme, exact labels, intervals, PDF+PNG+metadata export. |
| `runner.py` | Backend-neutral score, causal, carriage, cache, and figure orchestration. |

The focused ZINC/QM9 Colab workflow is
[`../../../experiments/methodology/grit_zinc_qm9_figures_colab.ipynb`](../../../experiments/methodology/grit_zinc_qm9_figures_colab.ipynb).
It keeps canonical caches immutable and writes supplemental diagnostics below a separate
task/seed namespace. Its GRIT raw-logit comparison is deliberately architecture-native:
node-only counterfactual versus actual relation/RRWP-conditioned logits, rather than
Graphormer's additive dot-versus-bias decomposition.

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

`ExecutionPolicy.graphs_per_batch` controls how many base graphs share an event forward. Batched
results are split back into graph-local sufficient statistics before donor/source/graph
aggregation. The setting and any OOM retries are recorded, but execution policy is deliberately
excluded from the scientific cache fingerprint. Clean Jacobians are computed once and reused
across the semantic and structural channels; clean-ablation baselines are likewise reused across
all head and family targets.

`ExecutionPolicy.replica_pair_budget` can additionally cap an approximate
`replicas * nodes^2` batch cost, and matching-output Jacobians use
`jacobian_output_chunk` batched VJPs. If the pinned PyTorch/PyG stack reports that an official
operator has no compatible vmap rule, the GraphBench backend rebuilds that clean forward and
retains exact sequential VJPs for the rest of the worker; unrelated autograd errors are not
suppressed. Each detached clean Jacobian is atomically persisted as soon as its graph completes,
then retained within the isolated worker and reused when scores and carriage address the same
discovery graph. A restarted worker loads these protected graph shards before computing any misses.
The selected engine is recorded in the model audits. These controls, OOM backoff, and heartbeat
frequency are execution-only and never enter the scientific fingerprint. After an OOM, subsequent
batches retain the successful smaller graph count instead of repeatedly retrying an oversized
batch. GraphBench heartbeats additionally sample device-wide utilization, VRAM use, and power
through `nvidia-smi`.

Independent HPC processes should call `run_worker` with the same complete multi-task/multi-seed
configuration and one selected task/seed. A worker writes only its `seed_<training-seed>` subtree;
it never writes shared task or root summaries. After every worker succeeds,
`finalize_cached_run` performs a model-free completeness and cache-contract check, renders every
seed, and becomes the sole writer of `population.json`, `protocol.json`, `audits.json`, and
`index.json`. This worker/finalizer boundary is required when seeds share an output root.

To add a future Graphormer dataset such as ZINC, register a `CanonicalTask` with
`backend_kind="graphormer"` and a `GraphormerTaskSpec`, then register its dataset builder with
`register_graphormer_dataset`. The builder returns evaluation and donor split views containing
`GraphormerGraph` records. No score, causal, carriage, or plotting estimator is duplicated.

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
        scores/{semantic,structural}/graph_*.pt
        carriage/{semantic,structural}/graph_*.pt
        causal/clean_ablation/*.pt
        causal/events/<channel>/graph_*/*.pt
      progress.jsonl
      figures/
        *.pdf
        *.png
        *.metadata.json
      figures.json
    population.json              # seed estimates; population CI only with >=3 seeds
```

The figures-only finalizer is seed-resumable. `figures.json` is written atomically only after one
seed finishes; a later finalizer reuses it only when every declared figure and metadata sidecar is
present and non-empty. A seed interrupted while plotting is rendered again, while earlier complete
seeds are skipped. Pass `--force` when intentionally regenerating every figure.

Numerical, invariance, and estimability audits report and continue by default: a breach is logged
once, merged by audit name with its worst observed value and repeat count, and written to the two
`audits.json` files (model checks also land in `canonical_audits.failures` of `model.json`).
`run(..., strict_audits=True)` restores fail-closed runs, raising `AuditError` on the first breach.
Cache fingerprints are unaffected by this switch, so strict and reporting runs share caches.

Every distance figure is drawn on a grouped **display axis** of at most
`FigureTheme.max_distance_points` columns (default 14, overridable per task through
`figure_overrides`). Near distances keep unit resolution and the tail widens dyadically, so a
121-column peptides axis becomes `0…9, 10-19, 20-39, 40-79, 80-121` instead of 122 illegible ticks;
a short-diameter task such as ZINC already fits and is left exactly as it was. Grouping is applied
to the cached per-graph and per-event sufficient statistics and the registered estimators are then
rerun on top — mass is summed, support-normalized panels are recomputed as summed contribution over
summed support inside each graph, and the interval is a fresh nested bootstrap over grouped
observations (a few seconds, from the cached event table). No figure ever re-derives a grouped
value from an already-aggregated one.

Additive quantities are displayed **per unit distance** once grouped, because a ten-wide tail group
accumulates ten columns of mass and would otherwise draw as a resurgence that is not in the data.
This affects the exact score panels and the attention profile, whose labels say so; support-
normalized panels are already width-invariant, and both are identities at unit resolution.

The registered 10-graph/50-pair reporting floor is then applied to the display columns. Each channel
gains a `{channel}_distance_column_support` figure showing supporting graphs, eligible pairs, and
the fraction of bootstrap replicates in which the column had no support at all; columns below the
floor are drawn in grey there and blanked in the heatmap and profile figures, with the suppressed
labels recorded in each figure's metadata. Grouping usually lifts the far tail over the floor, since
a group aggregates the support of every column in it. Cached arrays keep unit resolution throughout,
and a score cache written before any of this existed still gets all of it — everything needed is
recomputed from the cached graph support and event table, so only the figures phase needs rerunning.

`causal_validation_coordinates` prints each panel's rank correlation above it — `rho` with its
nested-bootstrap interval, the within-layer permutation `p`, and `n` — all of them already
estimated by the causal stage, so the figure reports the same numbers as `associations` rather than
recomputing anything. The `D_rel` panels use the active-head variants, since selectivity is only
defined there. No trend line is drawn: the statistic is a rank correlation, and an ordinary
least-squares line would show a different model from the one being reported. The grid also carries
the key it never had — the layer colourbar its point colours have always encoded, a marker for
heads below the activity floor, and a note that the whiskers are 95% nested percentile intervals.

`selectivity_vs_joint_sensitivity` now zooms to the observed `D_rel` support (including its
intervals) and shades the preregistered selectivity-equivalence region. This prevents a narrow
generalist cloud from being visually compressed against the mathematically possible `[-1, 1]`
range. `selectivity_regime_diagnostics` then shows the ordered active-head intervals, bootstrap
family Jaccard distributions, and interval-containment fractions. Family assignment is rerun in
all 2,000 coordinate-bootstrap draws; only inclusion probabilities and stability summaries are
cached, not the full transient draw tensor.

`causal_regime_summary` is the headline regime figure. Its first panel confirms that `J` predicts
clean and causal importance, its second is a forest plot of gross, necessity, rescue, and induction
family-by-channel interactions against the causal-equivalence region, and its third tests whether
the high-`J` central family responds to and rescues both channels after frozen reference scaling.
The machine-readable `regime_evidence` record makes the ternary call
(`confirmed_causal_specialisation`, `confirmed_entangled_generalist`, or `mixed_or_unresolved`)
from the preregistered rule rather than from figure inspection.

`causal_family_endpoints` and `causal_matched_control_endpoints` run their targets down a shared
vertical axis, one legible copy of the names rather than five rotated illegible ones, with panel
height following the target count. The matched-control figure shows the full-size frozen controls
only; the `control_prefix_*` ladder it used to enumerate — one bar category per (control, prefix
size) pair, around a hundred of them on a ten-layer model — is now drawn where a ladder belongs, as
a dotted reference line beneath its family's curve in `causal_cumulative_prefix_curves`. The
prefix figure now uses reference-scaled total and channel-contrast coordinates. Its total panels
show whether causal importance accumulates; its contrast panels show whether channel selectivity
accumulates and carry the causal-equivalence band.

Score distance heatmaps are head-resolved. `{channel}_score_distance_heatmaps` draws one row per
head, blocked by layer with layer 0 in the top block; the companion figure named with the extra
suffix `_row_normalised` repeats it with each head divided by its own total over the full distance
axis. Both come from one estimate; only the display differs, and each figure's metadata says which.
A score cache written before the head-resolved arrays existed still renders both — they are rebuilt
exactly from the cached per-graph contribution and support, so only the figures phase needs rerunning.

Each channel also emits `{channel}_carriage_profiles_event_normalised`. Its panels are titled
`Functional carriage (event-normalised)` and, where registered,
`Beneficial carriage (event-normalised)`. These are figure-only shape diagnostics derived from
the cached donor-event rows; raw carriage remains the primary scale-sensitive output.

Training checkpoints and dataset caches are read-only. Analysis caches bind the protocol
fingerprint, checkpoint SHA-256, task adapter, output representation and sigma, model geometry,
split IDs, event manifest, donor/source dose, bootstrap seed, `F_sens`, and the positive-beneficial
sign convention. The repository commit is stored as provenance, but it is not itself a scientific
validity field: moving the launcher to a newer commit does not invalidate an otherwise identical
cache. During a resumable canonical run, a genuinely incompatible or unreadable derived cache is
atomically moved under that seed's `cache/_stale/` tree and treated as a miss, so only affected
artifacts are recomputed. The archive includes a JSON sidecar with the old and expected contracts.
Standalone paper-artifact loaders remain read-only and fail closed on protocol or integrity
mismatches.

The v4 regime analysis changes the frozen central-family rule and records mismatch-adjusted
directional alignment fields, so a v3 causal cache cannot be relabelled as v4. Start v4 in a new
analysis output directory. Training checkpoints and dataset caches are still reused; once v4
scores and causal events exist, later figure revisions remain figures-only reruns.

The score cache also retains clean attention distance mass and frozen-family exact/support-
normalized score profiles as descriptive diagnostics. These never replace the transport score.
When at least three training seeds are run, `population.json` bootstraps seed-level summary and
association estimates without aligning head indices; with fewer seeds it records the individual
seed estimates and explicitly suppresses a seed-population interval.
