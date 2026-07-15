# Beneficial carriage: robustness investigation (synthetic)

Self-contained synthetic studies (numpy / torch, CPU, minutes to run) testing whether
beneficial semantic carriage can be measured robustly, and a k-hop over-squashing study.
Each script regenerates its own figures in `figures/` (`paper_style.py` = shared figure style).

Paper fragments (findings summary, experiment outlines E1–E6, appendix methodology of every
estimator) are in [`writeup.tex`](writeup.tex).

## Scripts → figures → finding

| script | figure | finding |
|---|---|---|
| `beneficial_carriage_synth.py` | `fig_scatter_grid.png`, `fig_neutral_leakage.png` | Round 1 (known manifold). Direct on-manifold-resample loss-diff matches ground truth everywhere; first-order `sign(r)·F` (current `benefit_sign` method) mis-flags spuriously-used nodes as *more* beneficial than genuine ones (4–5× neutral leakage). Choice of weight (hard/soft/raw) is irrelevant. |
| `beneficial_carriage_round2.py` | `fig_round2.png` | Break the known-manifold cheat: data-driven "matched" swaps + a trained MLP. Only the oracle stays good; data-driven finite drops to 0.4–0.6; the `var_ratio` health gate never trips. |
| `beneficial_carriage_round3.py` | `fig_round3.png` | Mechanism: estimator quality is a smooth monotone function of context-estimation error (environment identifiability). `var_ratio` stays flat across the whole collapse → blind to it. |
| `khop_oversquashing.py` | `fig_khop.png` | k-hop associative recall. Functional-carriage **reach < reachability wall kL** (compression eats the tail); GT niche = long-range band; skill: k=1..5 ≈ 0.0–0.24 vs global 0.99. |
| `beneficial_on_trained.py` | `fig_beneficial_trained.png` | Beneficial usage on trained models vs known ground truth. `B_direct` AUROC → **1.00** on GT model; `B_firstord` (current method) collapses to **0.54 (chance)** on the well-fit model; functional carriage ≠ benefit (over-flags distractors). |
| `beneficial_estimators.py` | `fig_estimators.png` | Practical estimator sweep vs sample size (GT + k=5), **on-manifold** regime. Many-swap `direct`≡`eg` win; `firstord` = chance on GT; single-eval `ig` is the cheap default; `perturb` < replacement. Sample budget scales with model misfit. |
| `off_manifold_estimators.py` | `fig_offmanifold.png` | **Tunable off-manifold risk** (checksum-penalty wrapper, dial κ). Matched-donor `direct` is **immune** (flat AUROC 1.0 ∀κ); `perturb_small` degrades slowest among donor-free (penalty ∝ κδ²); `ig`/`marginal` collapse by κ≈0.5–1; `firstord` dead throughout. Robustness ∝ 1/(input disruption). |

## Bottom line

- Keep **functional carriage** at full per-pair resolution (robust, label-free, mechanistic).
- Replace the per-pair **`benefit_sign` first-order shortcut** — it fails worst on well-trained models.
- Robust beneficial estimator = **on-manifold resample → direct held-out loss difference** (matched-environment donors in real data), gated by an environment-identifiability check (not `var_ratio`).
- Or the bulletproof coarse route: aggregate held-out task loss vs distance via receptive-field/attention ablation (model-side intervention keeps inputs on-manifold).
