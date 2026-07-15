# Official GRIT Peptides Slurm Setup

This setup trains official dense GRIT+RRWP and the parameter-matched 1-hop
sparse GRIT control on the LRGB Peptides tasks:

- `struct`: official config `configs/GRIT/peptides-struct-GRIT-RRWP.yaml`
- `func`: official config `configs/GRIT/peptides-func-GRIT-RRWP.yaml`

The dense runs use the pinned official GRIT config unchanged. The 1-hop control
copies the corresponding official config and changes only:

```yaml
gt.attn.full_attn: False
gt.attn.sparsity: one_hop
```

The masked RRWP encoder implementation is the same task-agnostic path used for
the ZINC 1-hop control.

## Submit Seed 42

On the HPC login node:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
git pull --ff-only
bash experiments/peptides/grit_official_slurm/hpc_seed42_launch.sh
```

This submits four independent one-GPU jobs:

- Peptides-struct dense GRIT
- Peptides-struct 1-hop GRIT
- Peptides-func dense GRIT
- Peptides-func 1-hop GRIT

Outputs are written under:

```text
/rds/user/jgg45/hpc-work/grit_peptides_seed42_results
```

Datasets/processed caches are separated by task and variant under:

```text
/rds/user/jgg45/hpc-work/grit_peptides_seed42_data
```

This avoids concurrent PyG cache writes when all jobs are submitted together.
