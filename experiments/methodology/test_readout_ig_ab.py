"""A/B test: does full-IG-through-the-readout carriage improve Step-0b fidelity to the measured
one-node-baseline output delta?

Background
----------
Carriage is C[i,j] = g_i . Delta h_i(j). By default g_i (the readout gradient) is frozen at the
clean value, so the reconstruction Sum_i C[i,j] only *approximates* the true single-node output
delta when the readout MLP is nonlinear. The `carriage_readout_ig` toggle re-evaluates g along the
IG path (full IG through the readout), making the reconstruction exact. This script measures, per
model, the Step-0b R^2 (predicted `Sum_i C[i,j]` vs the *measured* one-node-baseline delta y) for
BOTH the frozen-g carriage and the readout-IG carriage, so you can see whether the toggle improves
fidelity -- and by how much.

Environment
-----------
Run in the SAME Colab session as colab_zinc_main_procedure.py (deps installed, Drive mounted,
checkpoints present). Point --config at the config YAML the runner wrote (it holds the discovered
model checkpoint paths), e.g.:

    /content/drive/MyDrive/graph_specialisation_metrics/zinc_main_procedure_colab/configs/zinc_main_procedure_colab.yaml

Usage (Colab cell)
------------------
    !python /content/Graph-Specialisation-and-Metrics/experiments/methodology/test_readout_ig_ab.py \
        --config <that yaml path> --sample-graphs 8 --ig-steps 32

This reuses only forward passes + the existing carriage function; it does not write any artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _r2(measured: list[float], predicted: list[float]) -> float:
    import numpy as np

    y = np.asarray(measured, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 3:
        return float("nan")
    y, p = y[mask], p[mask]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - p) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-src", default="/content/Graph-Specialisation-and-Metrics/src")
    ap.add_argument("--config", required=True, help="Config YAML the runner wrote (holds checkpoint paths).")
    ap.add_argument("--sample-graphs", type=int, default=8)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args(argv)

    sys.path.insert(0, args.repo_src)
    import torch  # noqa: F401  (ensures torch is importable before the heavy imports)
    from graph_specialisation_metrics.main_procedure import discover_model_artifacts, load_config
    from graph_specialisation_metrics.grit_intervention_procedure import (
        carriage_ig,
        expanded_baseline,
        instantiate_official_models,
        mean_encoded_baseline,
        predict_scalar_from_encoded,
        select_baseline_graphs,
        select_graphs,
    )

    config = load_config(args.config, fast_dev_run=False, output_root=None, analysis_preset="full")
    discovery = [discover_model_artifacts(name, cfg) for name, cfg in config["models"].items()]
    models = instantiate_official_models(config, discovery)

    print(f"\nStep-0b A/B: predicted Sum_i C[i,j]  vs  measured one-node-baseline delta y")
    print(f"(ig_steps={args.ig_steps}, sample_graphs={args.sample_graphs})\n")
    print(f"{'model':24s} {'n_src':>6s} {'R2 frozen-g':>12s} {'R2 readout-IG':>14s} {'improvement':>12s}")
    print("-" * 72)
    for model in models:
        try:
            graphs = select_graphs(model.adapter, "test", args.sample_graphs, seed=args.seed)
            baseline = mean_encoded_baseline(
                model.adapter,
                select_baseline_graphs(model.adapter, "test", config, args.sample_graphs, seed=args.seed),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{model.name:24s} skipped ({type(exc).__name__}: {exc})")
            continue
        measured: list[float] = []
        pred_frozen: list[float] = []
        pred_rig: list[float] = []
        for gi, graph in enumerate(graphs):
            try:
                enc = model.adapter.encoded_node_states(graph).to(model.adapter.device)
                base = expanded_baseline(enc, baseline.to(model.adapter.device))
                clean_pred = float(predict_scalar_from_encoded(model.adapter, graph, enc).detach().cpu().item())
                c_frozen = carriage_ig(model.adapter, graph, baseline, steps=args.ig_steps, readout_ig=False)["carriage"]
                c_rig = carriage_ig(model.adapter, graph, baseline, steps=args.ig_steps, readout_ig=True)["carriage"]
                n = int(enc.size(0))
                for j in range(n):
                    pert = enc.detach().clone()
                    pert[j] = base[j]
                    pj = float(predict_scalar_from_encoded(model.adapter, graph, pert).detach().cpu().item())
                    measured.append(clean_pred - pj)                 # one-node baseline delta y (Step-0b convention)
                    pred_frozen.append(float(c_frozen[:, j].sum().item()))
                    pred_rig.append(float(c_rig[:, j].sum().item()))
            except Exception as exc:  # noqa: BLE001
                print(f"  {model.name} graph {gi}: failed ({type(exc).__name__}: {exc})")
        if len(measured) >= 3:
            r2f = _r2(measured, pred_frozen)
            r2r = _r2(measured, pred_rig)
            print(f"{model.name:24s} {len(measured):6d} {r2f:12.3f} {r2r:14.3f} {r2r - r2f:+12.3f}")
        else:
            print(f"{model.name:24s} insufficient data ({len(measured)})")
    print("\nHigher R2 = carriage tracks the measured single-node delta y better. readout-IG should")
    print("match or beat frozen-g; the residual gap for dense is the (informative) non-additivity.")


if __name__ == "__main__":
    main()
