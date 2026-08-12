# Layer-controlled joint-sensitivity correction

## Scope

This correction addresses the possibility that later transformer layers have systematically
larger clean-ablation effects, producing a pooled association between joint sensitivity `J` and
head-ablation impact even when there is little association among heads at the same depth.

The Colab covers:

- mixed synthetic task, Figure 4.5(a);
- PCQM4Mv2 Graphormer, Figure 5.6(a);
- dense ZINC GRIT, Figure 5.6(b); and
- dense QM9 GRIT, Figure 5.6(c).

GraphBench bipartite matching is not accessed from Colab because its completed cache is on HPC. It
will use the same estimator when revisited separately.

## Estimand

For trained model seed `s` and transformer layer `l`, compute Spearman correlation across heads:

```text
rho[s,l] = Spearman(J[s,l,:], ablation_impact[s,l,:])
```

The model-seed estimate is the equal-weight mean across its layers. The experiment estimate is the
equal-weight mean across trained seeds. Every per-seed/per-layer value is exported. This is not the
existing pooled effect with a within-layer permutation null, and it is not the layer-fixed-effect
OLS coefficient.

For equal-size, tie-free layers, the headline estimate is algebraically identical to ranking both
variables inside each seed/layer, centring those ranks, concatenating the residual ranks, and
correlating them. The manifest also reports that residual-rank diagnostic explicitly.

For the mixed synthetic experiment, the comparison table includes both the original correlation
pooled across every seed and layer and the equal-weight mean of the three seed-specific pooled
correlations. The latter isolates the effect of layer control from the separate choice to weight
trained seeds equally.

Molecular 95% intervals resample held-out molecules with replacement. Each bootstrap draw uses one
shared molecule draw for all heads, recomputes every head's mean impact, recomputes each layer's
Spearman correlation, and averages the layer estimates. Joint sensitivity and the trained heads
remain fixed, matching the inferential scope of the original Figure 5.6 intervals. The optional
randomisation test shuffles ablation impacts only among heads in the same trained-seed/layer
stratum.

## Exact dissertation caches

The default search starts at:

```text
/content/drive/MyDrive/graph_specialisation_metrics
```

It prefers these exact inputs:

```text
causal_specialisation_double_dissociation/cycle_dual_v2/
  tables/per_head_metrics.csv

graphormer_pcqm4mv2_causal/graphormer_pcqm4mv2/seed_0/
  cache/focused/core_tests.pt
  cache/focused/clean_ablation/graph_*.pt

canonical_methodology_v4_zinc_qm9/{zinc,qm9_gap_dense}/seed_42/
  cache/scores/raw.pt
  cache/causal/validation.pt
```

The expected dissertation contracts are 3 layers by 8 heads for each of three synthetic seeds,
12 by 32 heads and 128 held-out molecules for Graphormer, and 10 by 8 heads and 64 held-out
molecules for each dense GRIT task. The loader reconstructs cached mean impacts from per-molecule
rows and reproduces the original pooled correlation before accepting a cache.

Canonical wrapper files are opened through the repository's read-only integrity validator. Their
internal protocol and contract fingerprints must be self-consistent; paired score/causal caches
and core/shard caches must share the complete scientific contract, apart from the deliberately
stage-specific event-manifest hash and checkout provenance. Synthetic analysis shards, when
present, must carry the exact experiment version, analysis version, fingerprint, seed, and 3-by-8
geometry.

Discovery always inventories both known and shallow relocated roots. A candidate is selected only
after its schema, geometry, graph count, contracts, and reconstructed means validate, so an
incomplete cache in the conventional location cannot hide a complete relocated run.

Paper-population caches with 256 held-out molecules are listed as alternatives, not silently used
as replacements. Set `ALLOW_NON_DISSERTATION_FALLBACKS = True` only for an explicitly labelled
robustness rerun. If more than one equally preferred exact cache is found, set the corresponding
`CACHE_OVERRIDES` entry.

## Running

Open `experiments/methodology/layer_controlled_head_ablation_colab.ipynb` in a CPU Colab and run
all cells. Use `MODE = "inventory"` first when a Drive root may have moved. With the exact roots
available, use `MODE = "run"`.

The frontend installs only the base numerical and document-verification dependencies. It does not
install or load GRIT, Graphormer, PyG, RDKit, datasets, checkpoints, or CUDA components. The run
manifest records `cache_only: true` and `model_forwards: 0`.

## Outputs

Outputs are written separately from the source caches:

```text
layer_controlled_head_ablation_correction_v1/
  synthetic/figures/fig2a_sensitivity_impact_within_layer_ranks.{pdf,png,metadata.json}
  synthetic/figures/fig2a_sensitivity_impact.{pdf,png,metadata.json}
  graphormer_pcqm4mv2/figures/01_joint_sensitivity_head_ablation_within_layer_ranks.{pdf,png,metadata.json}
  graphormer_pcqm4mv2/figures/01_joint_sensitivity_head_ablation.{pdf,png,metadata.json}
  zinc/figures/01_joint_sensitivity_head_ablation_within_layer_ranks.{pdf,png,metadata.json}
  zinc/figures/01_joint_sensitivity_head_ablation.{pdf,png,metadata.json}
  qm9_gap_dense/figures/01_joint_sensitivity_head_ablation_within_layer_ranks.{pdf,png,metadata.json}
  qm9_gap_dense/figures/01_joint_sensitivity_head_ablation.{pdf,png,metadata.json}
  tables/pooled_vs_layer_controlled.csv
  tables/within_layer_correlations.csv
  run_manifest.json
```

Each task exports two intentionally different views. The primary `_within_layer_ranks` panel ranks
both variables separately inside each trained-seed/layer stratum using tie-aware
`(average_rank - 0.5) / n` percentiles. Its plotted cloud therefore removes between-layer location
shifts as well as reporting the corrected statistic. It retains the dissertation canvas,
typography, colours, markers, colorbar, legend, and annotation grammar, but uses linear percentile
axes. The original-stem companion retains the exact raw coordinates and dissertation axes, with
only the correlation annotation changed; it is available as the TeX drop-in and makes any layer
clustering visible. Task-specific directories prevent overwriting the originals. The manifest
binds each result to source paths and SHA-256 hashes, cache contracts, estimator and resampling
seeds, repository commit, coordinate view, and verified PDF/PNG dimensions.
