# ZINC Experiments

This folder contains the initial ZINC training component.

`notebooks/` preserves the original notebooks. `training/` contains extracted Python scripts for easier diffing, reuse, and automation.

Current models:

- GRIT
- Graphormer
- CSA
- GraphGPS

Planned:

- Exphormer

## Cache-only RRWP comparison

`analysis/zinc_cached_rrwp_comparison_colab.py` is the fast seed-42 comparison for
the six trained variants: local-RRWP 1-hop, global-RRWP 1-hop, 1-hop+VN, 2-hop,
2-hop+VN, and dense GRIT. It reads only canonical methodology artifacts:

```text
<canonical-root>/<task>/seed_42/model.json
<canonical-root>/<task>/seed_42/cache/scores/raw.pt
<canonical-root>/<task>/seed_42/cache/carriage/fields.pt  # optional
```

The score loader verifies the canonical task/checkpoint/split contract, and the
carriage loader verifies that it matches the score artifact. The run writes model
and head-level tables, exact per-head score-distance rows, pairwise profile
distances, and two focused figures. It does not construct a model, load ZINC, or
recompute scores.

For a mounted local canonical root, the same analysis can be run directly:

```bash
PYTHONPATH=src python -m graph_specialisation_metrics.zinc_cached_rrwp_comparison \
  --canonical-root /path/to/canonical_methodology_v4_zinc_qm9 \
  --output-dir /path/to/zinc_cached_rrwp_comparison \
  --train-seed 42
```
