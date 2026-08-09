# Peptides Drive cache map

```text
R = /content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models/peptides_func_struct_checkpoints/canonical_outputs
```

Each task below has three run directories: `R/<task>/seed_0`, `seed_1`, and `seed_2`.

| Model | `<task>` |
|---|---|
| Peptides-func dense | `peptides_func_dense` |
| Peptides-func 1-hop | `peptides_func_1hop` |
| Peptides-func 1-hop+VN | `peptides_func_1hop_vnode` |
| Peptides-func 2-hop | `peptides_func_2hop` |
| Peptides-func 2-hop+VN | `peptides_func_2hop_vnode` |
| Peptides-struct dense | `peptides_struct_dense` |
| Peptides-struct 1-hop | `peptides_struct_1hop` |
| Peptides-struct 1-hop+VN | `peptides_struct_1hop_vnode` |
| Peptides-struct 2-hop | `peptides_struct_2hop` |
| Peptides-struct 2-hop+VN | `peptides_struct_2hop_vnode` |

For every `<task>` and seed `<s>`:

```text
Scores:    R/<task>/seed_<s>/cache/scores/raw.pt
Carriage:  R/<task>/seed_<s>/cache/carriage/fields.pt

Clean shards:       R/<task>/seed_<s>/cache/clean_jacobians/graph_XXXXXX.pt
Semantic scores:    R/<task>/seed_<s>/cache/scores/semantic/graph_XXXXXX.pt
Structural scores:  R/<task>/seed_<s>/cache/scores/structural/graph_XXXXXX.pt
Semantic carriage:  R/<task>/seed_<s>/cache/carriage/semantic/graph_XXXXXX.pt
Structural carriage: R/<task>/seed_<s>/cache/carriage/structural/graph_XXXXXX.pt
```

Use `raw.pt` and `fields.pt` for downstream analysis; graph files are resumable shards.
