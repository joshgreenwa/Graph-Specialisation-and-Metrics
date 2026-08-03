# Graphormer PCQM4Mv2 causal head populations

## Purpose

This analysis turns the earlier focused causal diagnostics into two paper-facing claims:

1. whether score-defined semantic and structural head populations have different causal roles under
   semantic versus structural interventions; and
2. whether joint sensitivity `J` predicts clean-input head importance across all 384 heads.

The paper Colab uses mutually disjoint sets of 512 discovery, 512 causal-event, and 512
clean-ablation molecules. Selection and matching use discovery scores only. No restoration,
injection, necessity, or clean-ablation outcome enters the head gate. The semantic donor pool
remains disjoint and contains 2,000 additional molecules.

## Head populations and controls

The primary paper analysis requests 16 semantic/structural pairs. Within each direction, the
candidate pool is the strongest `3K` active heads beyond the registered `D_rel = +/-0.10` margin.
Hungarian assignment minimizes standardized absolute `J` distance with a small layer-distance
tie-break, and the `K` best-balanced pairs are retained. This produces 32 specialist heads by
default, rather than making a population claim from two or three examples.

Every selected specialist is also matched without replacement to a distinct active, non-selected
head. The null assignment prioritises `J`, then layer, with only a small preference for heads nearer
zero `D_rel`; it therefore does not force a poor `J` match merely to obtain a nominally neutral
control. The manifest records `J` mean differences standardized by the pre-match eligible-head
`J` standard deviation, mean and maximum `J` gaps, mean layer gap, exact-layer fraction, candidate
counts, and every head identity. At least 12 specialist
pairs and a complete one-control-per-specialist match are required; the analysis does not relax
these rules after seeing causal outcomes.

## Three primary causal tests

The first panel inserts each head's clean routed output into the intervened graph (restoration).
The second inserts its intervention-state output into the clean graph (injection). Both report the
direction-aligned movement beyond the same-source, same-tier, nearest-dose alternative-donor
activation control. The third independently ablates the head in clean and intervened runs and
reports the fraction of the intervention effect removed.

The semantic and structural populations are evaluated on both intervention channels. The
`J`-matched null family is shown for necessity. Confidence intervals jointly resample held-out molecules,
channel-specific sources, donors, and matched head pairs. These intervals characterize this one
trained checkpoint's selected head populations; they are not a substitute for training-seed
uncertainty.

## `J` versus clean ablation and the layer-adjusted coefficient

For every head, clean-ablation impact is the mean held-out movement
`||z - z_{-h}||_2`. The raw panel reports Spearman correlation. The adjusted panel globally
standardizes `J` and ablation impact, includes a fixed intercept for every layer, and plots the two
variables after those layer means have been removed. Its slope is exactly the reported
layer-adjusted standardized `beta`.

Thus `beta = 0.75` means that, comparing heads at the same layer, one global standard deviation
higher `J` predicts about `0.75` global standard deviations more clean-output movement. Adjustment
removes between-layer baseline differences; it does not prove causality, allow layer-specific
slopes, or generalize beyond the trained checkpoint. The paper preset's 95% interval resamples 512
held-out molecules while treating the 384 trained heads and discovery `J` values as fixed.

## Cache reuse and outputs

The population runner reads existing focused per-graph event shards and all-head clean-ablation
shards before scheduling work when the scientific contract is unchanged. The 512/512/512 paper
contract is stored under a separate Drive root, preserving the exploratory 128-molecule caches.
Dataset and checkpoint downloads remain shared. A new graph/channel shard contains reused rows plus
only the missing population heads. Styling-only reruns use `PHASE = "figures"` and do not load
Graphormer or PCQM4Mv2.

The Colab defaults target an A100-class runtime: eight clean-ablation graphs, 32 heads, and all
48 source/donor events are attempted per batch. Head batches recursively back off after a CUDA OOM;
on smaller GPUs, reduce `GRAPHS_PER_BATCH` first. `FORCE = False` keeps every completed per-graph
shard resumable across interrupted sessions.

Outputs are vector PDFs, 600-DPI PNGs, and JSON provenance sidecars under
`figures/focused_causal_population/`:

- `01_population_restoration_injection_necessity`;
- `02_J_vs_clean_ablation`; and
- `S01_population_head_selection`.
