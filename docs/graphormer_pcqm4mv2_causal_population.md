# Graphormer PCQM4Mv2 causal head populations

## Purpose

This analysis turns the earlier focused causal diagnostics into two paper-facing claims:

1. whether score-defined semantic and structural head populations have different causal roles under
   semantic versus structural interventions; and
2. whether joint sensitivity `J` predicts clean-input head importance across all 384 heads.

The paper Colab uses mutually disjoint sets of 256 discovery, 256 causal-event, and 256
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
raw direction-aligned movement, averaged across donor interventions. This directly tests the
aggregate claim behind `D_rel`: whether semantic-scoring heads respond more to semantic events and
structural-scoring heads respond more to structural events. The third independently ablates the
head in clean and intervened runs and reports the fraction of the intervention effect removed.

Each panel also reports one "correct-pairing advantage": semantic heads on semantic rather than
structural events, combined with structural heads on structural rather than semantic events. Its
bootstrap interval directly states whether the two score-defined families have different causal
roles. A separate retained figure subtracts the same-source, same-tier, nearest-dose
alternative-donor response. That stricter donor-specific analysis is a robustness check, not the
primary test of aggregate semantic/structural specialization.

A complementary continuous test avoids relying only on the two specialist labels. For every
selected or matched head, causal preference is its semantic-event effect minus its
structural-event effect. The analysis tests whether discovery `D_rel` predicts this preference for
restoration and injection. It reports raw Spearman correlation and a standardized regression
coefficient that also accounts for the head's mean absolute causal response and layer. This
controls overall responsiveness without dividing by a potentially near-zero response. Intervals
jointly resample intervention events and the matched four-head blocks.

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
slopes, or generalize beyond the trained checkpoint. The paper preset's 95% interval resamples 256
held-out molecules while treating the 384 trained heads and discovery `J` values as fixed.

## Cache reuse and outputs

The population runner reads existing focused per-graph event shards and all-head clean-ablation
shards before scheduling work when the scientific contract is unchanged. The 256/256/256 paper
contract is stored under a separate Drive root, preserving the exploratory 128-molecule caches.
Dataset and checkpoint downloads remain shared. A new graph/channel shard contains reused rows plus
only the missing population heads. Styling-only reruns use `PHASE = "figures"` and do not load
Graphormer or PCQM4Mv2. To add the continuous causal-preference result to a cache made by an older
notebook revision, run once with `PHASE = "all"`; the upgrade rebuilds bootstrap summaries from the
event shards with zero model forwards.

The Colab defaults target an 80 GB A100-class runtime: 16 clean-ablation graphs, 64 heads, and all
48 source/donor events are attempted per batch. Graph-score batches and head batches recursively
back off after a CUDA OOM; on smaller GPUs, reduce `GRAPHS_PER_BATCH` first. `FORCE = False` keeps
every completed per-graph shard resumable across interrupted sessions.

Execution progress is printed to the Colab output and appended to `progress.jsonl`. It records
stage transitions, score batches, cache hits and misses, causal graph-channel and head completion,
clean-ablation graph/head completion, throughput and ETA, bootstrap draws, CUDA allocation peaks,
and 30-second device-wide GPU utilization, VRAM, and power heartbeats.

Outputs are vector PDFs, 600-DPI PNGs, and JSON provenance sidecars under
`figures/focused_causal_population/`:

- `01_population_raw_restoration_injection_necessity` (primary);
- `01b_correct_pairing_advantage` (direct matching-versus-crossed summary);
- `01_population_restoration_injection_necessity` (retained mismatch-adjusted robustness check);
- `02_J_vs_clean_ablation`;
- `03_Drel_vs_causal_preference`; and
- `S01_population_head_selection`.
