# Checkpoints

Store checkpoint documentation here.

Large checkpoint binaries are ignored by git by default. For each run, document:

- model architecture;
- task and dataset split;
- training script or notebook;
- command-line arguments;
- seed;
- final validation and test metrics;
- external artifact location, if the checkpoint is stored outside git.

## Registered GRIT receptive-field controls

The canonical task registry reconstructs each training patch before loading its checkpoint. The
focused ZINC/QM9 figure notebook uses the same registrations; no separate checkpoint mapping is
required when these Drive folders retain their training layout.

| Canonical task | Parameters |
| --- | ---: |
| `zinc_1hop` | 473,473 |
| `zinc_1hop_vnode` | 473,537 |
| `zinc_2hop` | 473,473 |
| `zinc_2hop_vnode` | 473,537 |
| `qm9_gap_1hop` | 472,769 |
| `qm9_gap_1hop_vnode` | 472,833 |

The exact Drive locations remain single-sourced in the canonical task registry and are printed by
the figure notebook's preflight after Drive is mounted.

Standard GraphGym `results/**/ckpt/*.ckpt` files are discovered first. The ZINC k-hop/VN runner's
`results/_recovery_checkpoints/**/{best,latest}.ckpt` layout is also supported, with `best.ckpt`
preferred. A completed canonical `model.json` remains authoritative for figure-time checkpoint
identity and SHA verification.
