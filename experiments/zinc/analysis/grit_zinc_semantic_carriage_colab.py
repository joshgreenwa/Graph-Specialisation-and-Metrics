#!/usr/bin/env python3
"""
Colab runner: semantic-intervention carriage for the official GRIT ZINC checkpoint.

This script loads a checkpoint produced by ``train_grit_zinc_official_colab.py`` from
Google Drive and measures **semantic carriage**, **functional carriage F(d)** and
**beneficial carriage B(d)** exactly as specified in the dissertation, Chapter 3
(Sections 3.2-3.3, pp. 29-34). Structural interventions are deliberately out of scope;
only the semantic (content) intervention of Definition 3.2.1 is implemented.

Environment setup mirrors ``train_grit_zinc_official_colab.py``: same official repo
(pinned commit), same official ZINC RRWP config, same dependency set, same Python-3.12
compatibility fixes, same 473,473-parameter guard. Unlike the training runner (which
launches ``main.py`` as a subprocess), the analysis runs entirely IN-PROCESS: a pasted
Colab cell that calls ``main([...])`` has no ``__file__`` to relaunch, so the
compatibility fixes are applied directly (``apply_compat_patches``) before GRIT is
imported, and GRIT is driven from within this process.

======================================================================================
METHODOLOGY (dissertation Ch. 3, pp. 29-34) -- what this file implements, term by term
======================================================================================

Setup and notation (S3.2.1)
    A graph carries a semantic part X = (x_1..x_n) (node content) and a structural part
    S (topology E plus any encoding computed from it, e.g. RRWP). The model predicts
    yhat = f(X, S) via final-layer node states h^L_i and a permutation-invariant readout
    rho:  yhat = rho(h^L_1, ..., h^L_n).  The readout gradient is

        g_i = d yhat / d h^L_i                                                  (3.1)

    d(i, j) is the shortest-path (hop) distance. Node j (perturbed) is the *source*;
    node i (read under the perturbation) is the *carrier*.

    -> For official GRIT on ZINC:
         X            = data.x[:, 0], the integer atom type (TypeDictNode, 21 types).
         S            = edge_index, edge_attr (bond types), and RRWP
                        (data.rrwp / rrwp_index / rrwp_val / log_deg / deg).
         h^L_i        = batch.x after the 10th GritTransformerLayer, i.e. the tensor
                        entering post_mp. Dim 64.
         rho          = SANGraphHead: global_add_pool over nodes, then an MLP
                        (64->32->16->1). NOTE: grit_model.py constructs the head as
                        GNNHead(dim_in, dim_out) without passing L, so SANGraphHead's
                        default L=2 is used and cfg.gnn.layers_post_mp=3 is inert
                        (the config yaml says as much: "Not used when gnn.head: san_graph").
                        The parameter budget confirms this independently:
                          encoders = 21*64 (node emb) + 4*64 (edge emb)
                                   + 21*64 (RRWP abs) + 21*64 (RRWP rel) = 4,288
                          head L=2 = 2,080 + 528 + 17                    = 2,625
                          473,473 - 4,288 - 2,625 = 466,560 = 10 * 46,656 exactly.
                        With L=3 the remainder is 466,432, which is not divisible by
                        the 10 GRIT layers -- so L=2 is the only consistent reading.
         g_i          = obtained by autograd of yhat w.r.t. the captured h^L tensor.
                        Because pooling is `add`, g_i is mathematically identical for
                        every node i of a graph; this is VERIFIED at runtime, not assumed.

Semantic intervention (Def 3.2.1)
        X_{j->xt} = (x_1, ..., x_{j-1}, xt, x_{j+1}, ..., x_n),   S unchanged     (3.2)

    -> Implemented by rewriting a single entry of the batched atom-type tensor,
       batch.x[row_of(j), 0] = donor_type. Everything structural is left untouched.
       Because GRIT's RRWP is a pre-transform of edge_index alone and the config sets
       posenc_RRWP.add_node_attr: False, the structural encoding is provably invariant
       to this edit. The runtime check `--verify` asserts, per intervened graph, that
       edge_index / edge_attr / rrwp / rrwp_index / rrwp_val are bit-identical to clean
       and that x differs in exactly one entry.

Donor swap (Def 3.2.2) and donor averaging (S3.2.2)
        xt = x'_{j'},  G' ~ script-G,  j' in V(G')                               (3.3)

    -> Donors are real atom types sampled uniformly over the nodes of *other graphs*
       in the same dataset split ("the dataset from which G is drawn"), which keeps the
       perturbed graph on-manifold. K donors D_j = {xt^(1)..xt^(K)} are sampled
       INDEPENDENTLY PER SOURCE j (the dissertation indexes the donor set by j) and
       averaged, marginalising out donor identity so that what remains is the
       contribution of x_j itself.

       Donors are sampled uniformly from the empirical node distribution WITHOUT
       excluding donors whose atom type equals x_j. That is deliberate and required:
       the estimator is an expectation over the marginal content distribution, so
       same-type donors are genuine zero-effect draws, not a bug. (ZINC is ~70% carbon,
       so this fraction is large; it is reported as `donor_noop_fraction`. Those exact
       no-ops double as a free correctness probe -- see VERIFICATION below.)

Semantic carriage (Def 3.3.1) and its donor-swap estimator (3.5)
        C[i, j] = g_i^T . dh_i(j),  dh_i(j) = h^L_i(X, S) - h^L_i(X_{j->xt}, S)  (3.4)

        C_swap[i, j] = (1/K) sum_k  g_i^T [ h^L_i(X,S) - h^L_i(X_{j->xt^(k)}, S) ]  (3.5)

    -> Note the direction: CLEAN minus SWAPPED. g_i is taken at the CLEAN input and is
       NOT re-evaluated per donor (Eq. 3.5 does not index g by k; Remark 3.3.1 states
       the benefit direction is read off "the loss geometry at the clean, correctly-
       labeled input"). Donors are averaged BEFORE any absolute value or sign is taken.

Functional carriage (Def 3.3.2)
        F[i, j] = |C[i, j]|                                                      (3.6)
        F(d)    = mean over {(i,j) : d(i,j) = d} of |C[i, j]|                    (3.7)

    -> Label-free. abs() is applied to the donor-AVERAGED C (Eq. 3.6 is a function of
       C, and C is the estimator of 3.5), then the mean is taken over pairs.

Beneficial carriage (Def 3.3.3, made exact across the |.| kink)
    The dissertation's linearized form is B[i,j] = sign(yhat_clean - y) . C[i,j]  (3.8),
    whose column sum equals L_clean - L_base only to first order and only when the
    residual does not change sign. We use the EXACT per-source loss change instead,
    attributed to carriers by their share of the output change:

        L_clean = |yhat_clean - y|
        dL_j    = L_clean - mean_k |yhat_swap(j,k) - y|          per source j  [MAE units]
        B[i,j]  = dL_j . ( C[i,j] / sum_i C[i,j] )               carrier share
        => sum_i B[i,j] = dL_j   EXACTLY.

    Sign is unchanged and reads the same way (units are MAE):
        B < 0  -> transport REDUCED the error   : beneficial
        B > 0  -> transport INCREASED the error : adverse
        B ~ 0  -> moved the output, not the error: dispensable / redundant

    Two subtleties that make this correct rather than merely exact:
      * Donor-average the LOSS, not the prediction. The |.| kink makes E|.| != |E.|
        precisely when donor variance is large (i.e. near a residual sign crossing), so
        averaging yhat over donors and then taking |.| would misstate benefit. dL_j uses
        mean_k |yhat_swap - y|, the actual post-swap loss.
      * Because it uses the actual post-swap loss, this B SEES that corrupting a node's
        own content is catastrophic: the swap sends the loss far up, so dL_j = L_clean -
        L_swap is a large NEGATIVE number, i.e. the content was strongly beneficial (its
        removal hurt a lot). The clean-point linearization sign(yhat_clean - y).C cannot
        capture this magnitude across the kink. The carrier split C[i,j]/sum_i C[i,j] is a
        signed share of the output change; sources that move nothing (sum_i C[i,j] ~ 0,
        hence dL_j ~ 0) get a zeroed column via an eps guard, which is well-defined.

    Per Remark 3.3.1 we still never score the perturbed graph against the clean label to
    make a per-pair CLAIM in the linearized sense; here the post-swap loss enters only as
    the exact scalar dL_j being decomposed, and the transport dh_i(j) (via C) is what
    distributes it across carriers. B<0/>0/~0 remains a statement about the trained
    model's behaviour, not a ground-truth causal claim about the task.

B_far(k)  [requested deliverable]
        B_far(k) = sum over {(i, j) : d(i, j) > k} of B[i, j]

    -> Per graph (so it stays in MAE units and telescopes with the sum rule), then
       averaged over graphs with a bootstrap CI. Negative = beneficial.

======================================================================================
VERIFICATION (every one of these runs by default; --verify off to skip the costly one)
======================================================================================
  1. Parameter count is exactly 473,473 (GRIT paper Table 9 ZINC row).
  2. Upstream ZINC RRWP config matches the official values (same table as the trainer).
  3. Checkpoint loads with strict=True and zero missing/unexpected keys; epoch reported.
  4. Test MAE is recomputed from the loaded checkpoint over the full test split and
     compared against the GRIT paper's ~0.059. A bad load shows up here immediately.
  5. Sum-pooling readout: max_i |g_i - g_1| is reported. It must be ~0; if it is not,
     the head is not pooling->MLP and the g_i story would need revisiting.
  6. Batch invariance: h^L for the clean graph computed alone vs. computed inside a
     large replica batch must agree to tolerance. This is what makes it legitimate to
     evaluate n*K perturbed replicas in one forward pass (requires model.eval(), since
     gt.batch_norm=True and gt.attn_dropout=0.2 would otherwise couple replicas).
  7. No-op donors: whenever a sampled donor's atom type equals x_j, the swap is the
     identity, so dh must be exactly 0. Asserting this confirms the intervention is
     wired to the intended node and that nothing else in the batch drifts.
  8. Structure invariance: only x changes under the intervention (bit-identical
     edge_index / edge_attr / rrwp / rrwp_index / rrwp_val).
  9. Additivity: sum_i C[i, j] vs (yhat_clean - mean_k yhat_swap(j,k)), reported as
     Pearson r and OLS slope. Audits that the carrier "share" C[i,j]/sum_i C[i,j]
     divides a meaningful output change (the head MLP is nonlinear, so r<1 is expected).
 10. Connectivity: unreachable pairs (d = inf) are counted; ZINC molecules are
     connected so this should be 0. Any such pairs are excluded from all
     distance-indexed statistics and loudly reported.
 11. Beneficial exactness: max_j |sum_i B[i,j] - dL_j| must be ~0. B is constructed so
     each source column sums to the exact per-source loss change; this asserts the share
     attribution and the loss donor-average are wired correctly (aborts above --tol).

Suggested Colab usage:

    from grit_zinc_semantic_carriage_colab import main
    main(["--skip-install"])          # same runtime as a finished training run
    main([])                          # fresh runtime

Outputs (default /content/drive/MyDrive/grit_zinc_official/carriage_analysis):
    figures/fig_semantic_carriage_Fd_Bd.png        F(d) and B(d)
    figures/fig_semantic_carriage_Bfar.png         B_far(k), MAE units
    figures/fig_semantic_carriage_diagnostics.png  counts, per-distance MAE mass,
                                                   additivity audit, per-graph F(d)
    carriage_pairs.npz                             every (graph, i, j, d, C, B)
    carriage_summary.json                          all curves + every check above
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Mapping, Sequence


# ----------------------------------------------------------------------------------
# Provenance constants -- identical to train_grit_zinc_official_colab.py
# ----------------------------------------------------------------------------------
OFFICIAL_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_CFG = "configs/GRIT/zinc-GRIT-RRWP.yaml"
OFFICIAL_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
EXPECTED_ZINC_GRIT_RRWP_PARAMS = 473_473

# GRIT paper (Table 2) ZINC-subset test MAE for GRIT+RRWP.
PAPER_ZINC_TEST_MAE = 0.059

EXPECTED_CFG_VALUES = {
    ("metric_best",): "mae",
    ("metric_agg",): "argmin",
    ("dataset", "format"): "PyG-ZINC",
    ("dataset", "name"): "subset",
    ("dataset", "task"): "graph",
    ("dataset", "task_type"): "regression",
    ("dataset", "node_encoder"): True,
    ("dataset", "node_encoder_name"): "TypeDictNode",
    ("dataset", "node_encoder_num_types"): 21,
    ("dataset", "edge_encoder"): True,
    ("dataset", "edge_encoder_name"): "TypeDictEdge",
    ("dataset", "edge_encoder_num_types"): 4,
    ("posenc_RRWP", "enable"): True,
    ("posenc_RRWP", "ksteps"): 21,
    ("posenc_RRWP", "add_identity"): True,
    # add_node_attr=False is what makes RRWP provably invariant to a content swap.
    ("posenc_RRWP", "add_node_attr"): False,
    ("posenc_RRWP", "add_inverse"): False,
    ("model", "type"): "GritTransformer",
    ("model", "loss_fun"): "l1",
    # graph_pooling=add is what makes the readout gradient g_i shared across nodes.
    ("model", "graph_pooling"): "add",
    ("gt", "layer_type"): "GritTransformer",
    ("gt", "layers"): 10,
    ("gt", "n_heads"): 8,
    ("gt", "dim_hidden"): 64,
    ("gt", "dropout"): 0.0,
    ("gt", "attn_dropout"): 0.2,
    ("gt", "layer_norm"): False,
    ("gt", "batch_norm"): True,
    ("gt", "update_e"): True,
    ("gt", "attn", "clamp"): 5.0,
    ("gt", "attn", "act"): "relu",
    ("gt", "attn", "full_attn"): True,
    ("gt", "attn", "edge_enhance"): True,
    ("gt", "attn", "O_e"): True,
    ("gt", "attn", "norm_e"): True,
    ("gt", "attn", "fwl"): False,
    ("gnn", "head"): "san_graph",
    ("gnn", "layers_pre_mp"): 0,
    ("gnn", "dim_inner"): 64,
    ("gnn", "act"): "relu",
    ("gnn", "dropout"): 0.0,
}


class CommandError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd))
    log(f"\n[cmd] {printable}")
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise CommandError(f"Command failed with exit code {proc.returncode}: {printable}")
    return proc


# ==================================================================================
# Stage 1: environment (mirrors train_grit_zinc_official_colab.py)
# ==================================================================================

def apply_compat_patches() -> None:
    """In-process equivalent of the training runner's sitecustomize shim.

    train_grit_zinc_official_colab.py runs GRIT's main.py as a SUBPROCESS and injects a
    `sitecustomize.py` via PYTHONPATH, which Python imports at interpreter startup. This
    analysis instead runs GRIT *inside the current process*: a Colab cell that pastes this
    file and calls main() has no __file__, so there is no script path to hand to a
    subprocess. The same fixes are therefore applied directly here.

    Fixes, all for Python 3.12 / modern Colab running GRIT-era dependencies:
      1. pkgutil.ImpImporter and FileFinder.find_module were removed in 3.12; old
         pkg_resources (reached via Lightning/PyG) still references them.
      2. PyTorch >= 2.6 flipped torch.load's default to weights_only=True, which breaks
         PyG 2.2's saved ZINC Data objects and GraphGym's checkpoints. Both are locally
         generated and trusted here. Only unspecified calls are patched.
      3. scikit-learn >= 1.6 removed mean_squared_error(..., squared=False); GRIT's
         logger still calls it.

    Idempotent, and a no-op where the runtime does not need the fix. Does not alter any
    GRIT model, dataset, optimizer or config logic.
    """
    import importlib.machinery
    import pkgutil
    import site
    import sysconfig

    # Prefer pip-installed packages in /usr/local over Debian's /usr/lib copy, which can
    # ship an old pkg_resources that shadows setuptools and crashes on removed APIs.
    preferred = []
    for _p in list(site.getsitepackages()) + [
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
    ]:
        if _p and os.path.isdir(_p) and "/usr/local/" in _p and _p not in preferred:
            preferred.append(_p)
    for _p in reversed(preferred):
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)

    if not hasattr(pkgutil, "ImpImporter"):
        class _CompatImpImporter:  # pragma: no cover - compatibility only
            pass
        pkgutil.ImpImporter = _CompatImpImporter

    if not hasattr(importlib.machinery.FileFinder, "find_module"):
        def _compat_find_module(self, fullname, path=None):
            spec = self.find_spec(fullname)
            return None if spec is None else spec.loader
        importlib.machinery.FileFinder.find_module = _compat_find_module

    # Drop a stale Debian pkg_resources so a later import prefers the /usr/local copy.
    _mod = sys.modules.get("pkg_resources")
    _mod_file = str(getattr(_mod, "__file__", "") or "") if _mod is not None else ""
    if _mod_file.startswith("/usr/lib/python3/dist-packages/"):
        del sys.modules["pkg_resources"]

    try:
        import torch
        if getattr(torch.load, "__name__", "") != "_compat_torch_load":
            _orig_torch_load = torch.load

            def _compat_torch_load(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig_torch_load(*a, **kw)

            torch.load = _compat_torch_load
            log("[compat] torch.load defaults to weights_only=False for trusted local files.")
    except Exception as exc:  # noqa: BLE001
        log(f"[compat-warning] could not patch torch.load: {exc}")

    try:
        import inspect
        import numpy as _np
        import sklearn.metrics as _sk_metrics

        if "squared" not in inspect.signature(_sk_metrics.mean_squared_error).parameters:
            _orig_mse = _sk_metrics.mean_squared_error

            def _compat_mean_squared_error(
                y_true, y_pred, *, sample_weight=None,
                multioutput="uniform_average", squared=True,
            ):
                mse = _orig_mse(y_true, y_pred, sample_weight=sample_weight,
                                multioutput=multioutput)
                return mse if squared else _np.sqrt(mse)

            _sk_metrics.mean_squared_error = _compat_mean_squared_error
            log("[compat] restored legacy sklearn mean_squared_error(squared=False).")
    except Exception:  # noqa: BLE001
        pass


def prepare_inprocess(repo_dir: Path) -> None:
    """Make `import grit` work in THIS process and match main.py's cwd expectations."""
    import importlib

    apply_compat_patches()
    # `pip install -e` writes a .pth that is only read at interpreter startup, so an
    # editable install performed during this same process may not be importable yet.
    # Putting the repo root on sys.path makes `import grit` work either way.
    importlib.invalidate_caches()
    p = str(repo_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    # GRIT's --cfg path is relative, and main.py sets cfg.work_dir = os.getcwd().
    os.chdir(p)
    log(f"[compat] In-process GRIT: sys.path[0]={sys.path[0]} | cwd={os.getcwd()}")


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(mount_point: Path) -> None:
    if in_colab():
        from google.colab import drive  # type: ignore
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(str(mount_point), force_remount=False)
    else:
        log("[drive] google.colab is unavailable; assuming a non-Colab run.")
        log(f"[drive] Using local path instead: {mount_point}")
        mount_point.mkdir(parents=True, exist_ok=True)


def pip_install(args: Sequence[str]) -> None:
    run_cmd([sys.executable, "-m", "pip", "install", *args])


def install_dependencies(args: argparse.Namespace) -> None:
    """Identical dependency set to the training runner, plus matplotlib for figures."""
    log("\n[deps] Installing Python dependencies. This may take a while on a fresh runtime.")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed/importable in this runtime.") from exc

    torch_version = str(torch.__version__).split("+")[0]
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = ("cu" + cuda_version.replace(".", "")) if cuda_version else "cpu"

    log(f"[deps] Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {cuda_version}")
    pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    log(f"[deps] PyG wheel index: {pyg_wheel_url}")

    for package in ("pyg-lib", "torch-spline-conv"):
        proc = run_cmd(
            [sys.executable, "-m", "pip", "install", package, "-f", pyg_wheel_url],
            check=False,
        )
        if proc.returncode != 0:
            log(f"[deps-warning] {package} wheel unavailable for this stack; continuing "
                f"because GRIT ZINC RRWP does not require it.")

    for package in ("torch-scatter", "torch-sparse", "torch-cluster"):
        proc = run_cmd(
            [sys.executable, "-m", "pip", "install", package, "-f", pyg_wheel_url],
            check=False,
        )
        if proc.returncode != 0:
            raise CommandError(
                f"Required PyG extension {package!r} failed to install from {pyg_wheel_url}."
            )

    pip_install([f"torch-geometric=={args.pyg_version}"])
    pip_install([
        "yacs==0.1.8",
        "pytorch-lightning==1.9.5",
        "torchmetrics==0.9.1",
        "opt_einsum>=3.3",
        "tensorboardX>=2.6,<2.7",
        "ogb==1.3.6",
        "wandb>=0.16,<0.18",
        "pyyaml>=6.0",
        "scikit-learn>=1.0",
        "scipy>=1.9",
        "networkx>=2.8",
        "matplotlib>=3.6",
    ])


def clone_or_update_repo(repo_dir: Path, repo_url: str, branch: str,
                         commit: str | None, force_fresh: bool) -> str:
    if force_fresh and repo_dir.exists():
        log(f"[repo] Removing existing repo: {repo_dir}")
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        run_cmd(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])
    else:
        log(f"[repo] Existing repo found: {repo_dir}")
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", branch], cwd=repo_dir)
        run_cmd(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)

    if commit:
        log(f"[repo] Pinning GRIT to commit: {commit}")
        run_cmd(["git", "checkout", commit], cwd=repo_dir)

    resolved = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    log(f"[repo] Using GRIT commit: {resolved}")
    return resolved


def get_nested(d: Mapping, path: Sequence[str]):
    cur = d
    for key in path:
        cur = cur[key]
    return cur


def _semantic_cfg_equal(actual, expected) -> bool:
    """Compare YAML config values semantically (1e-5 may parse as str or float)."""
    if actual == expected:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= max(1e-12, 1e-9 * abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return False


def validate_official_config(repo_dir: Path, allow_drift: bool) -> None:
    import yaml

    cfg_path = repo_dir / OFFICIAL_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Official config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errors: List[str] = []
    for path, expected in EXPECTED_CFG_VALUES.items():
        try:
            actual = get_nested(cfg, path)
        except Exception:
            errors.append(f"missing {'.'.join(path)}; expected {expected!r}")
            continue
        if not _semantic_cfg_equal(actual, expected):
            errors.append(f"{'.'.join(path)} = {actual!r}; expected {expected!r}")

    if errors:
        msg = "Official ZINC config does not match the expected GRIT ZINC+RRWP setup:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        if allow_drift:
            log("[config-warning] " + msg)
        else:
            raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")

    log("[config] Official GRIT ZINC+RRWP config validated.")
    log("[config] posenc_RRWP.add_node_attr=False  -> RRWP depends on topology only, so a "
        "content swap provably leaves S unchanged (Def 3.2.1).")
    log("[config] model.graph_pooling=add          -> readout is sum-pool + MLP, so g_i is "
        "shared across nodes (verified at runtime).")


def find_checkpoint(results_root: Path, explicit: str | None) -> tuple[Path, int]:
    """Locate the checkpoint to analyse.

    GraphGym writes <out_dir>/<cfg_name>-<name_tag>/<seed>/ckpt/<epoch>.ckpt. The
    official ZINC config uses ckpt_best=True + ckpt_clean=True, so the surviving file
    is both the LAST-saved and the BEST-validation checkpoint.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--ckpt does not exist: {p}")
        epoch = int(p.stem) if p.stem.isdigit() else -1
        log(f"[ckpt] Using explicitly requested checkpoint: {p} (epoch={epoch})")
        return p, epoch

    if not results_root.exists():
        raise FileNotFoundError(
            f"Results root not found: {results_root}\n"
            f"Point --drive-dir at the same directory the training runner used."
        )

    candidates = sorted(results_root.glob("**/ckpt/*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No *.ckpt found under {results_root}/**/ckpt/.\n"
            f"Has training saved a checkpoint yet? (train.enable_ckpt must be True.)"
        )

    log(f"[ckpt] Found {len(candidates)} checkpoint file(s) under {results_root}:")
    for c in candidates:
        log(f"[ckpt]   {c}  (mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c.stat().st_mtime))})")

    # Newest ckpt directory, then the highest epoch inside it => "last saved".
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    ckpt_dir = newest.parent
    in_dir = [p for p in candidates if p.parent == ckpt_dir]
    numeric = [p for p in in_dir if p.stem.isdigit()]
    chosen = max(numeric, key=lambda p: int(p.stem)) if numeric else newest
    epoch = int(chosen.stem) if chosen.stem.isdigit() else -1

    log(f"[ckpt] Selected last checkpoint: {chosen} (epoch={epoch})")
    log("[ckpt] (official ZINC config uses ckpt_best+ckpt_clean, so this is also the "
        "best-validation checkpoint)")
    return chosen, epoch


def print_environment_summary(drive_dir: Path, repo_dir: Path, commit: str) -> None:
    log("\n[env] Runtime summary")
    log(f"  platform: {platform.platform()}")
    log(f"  python:   {sys.version.replace(os.linesep, ' ')}")
    try:
        import torch
        log(f"  torch:    {torch.__version__}")
        log(f"  cuda:     {torch.version.cuda} | available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            log(f"  gpu:      {props.name} | {props.total_memory / 1024**3:.1f} GiB")
    except Exception as exc:
        log(f"  torch:    not importable ({exc})")
    log(f"  repo:     {repo_dir}")
    log(f"  commit:   {commit}")
    log(f"  drive:    {drive_dir}")


# ==================================================================================
# Core estimator (module level so it is unit-testable without a GPU or GRIT)
# ==================================================================================

def carriage_from_states(h_clean, h_swap, g, num_sources: int, num_donors: int):
    """Donor-swap semantic carriage estimator, dissertation Eq. 3.4 / 3.5.

        C_swap[i, j] = (1/K) sum_k  g_i^T [ h^L_i(X, S) - h^L_i(X_{j->xt^(k)}, S) ]

    Args:
        h_clean: [n, m]       h^L_i(X, S), the clean final-layer states.
        h_swap:  [S*K, n, m]  h^L_i under each intervention. Replica r = j*K + k,
                              matching Batch.from_data_list's contiguous stacking.
        g:       [n, m]       g_i = d yhat / d h^L_i, evaluated at the CLEAN input.
                              Eq. 3.5 does not index g by k, and Remark 3.3.1 takes the
                              benefit direction from the clean, correctly-labeled input.
        num_sources: S (one source per node, so S == n).
        num_donors:  K, the number of donors averaged over.

    Returns:
        [n, n] tensor C[i, j]: carrier i (rows) x source j (columns).

    Sign convention: dh is CLEAN minus SWAPPED, exactly as written in Eq. 3.4. The
    donor average is taken BEFORE any abs() (Eq. 3.6) or sign() (Eq. 3.8), because
    both are defined as functions of the estimator C, not of the per-donor terms.
    """
    import torch

    S, K = int(num_sources), int(num_donors)
    n, m = h_clean.shape
    if h_swap.shape != (S * K, n, m):
        raise ValueError(f"h_swap {tuple(h_swap.shape)} != {(S * K, n, m)}")
    if g.shape != (n, m):
        raise ValueError(f"g {tuple(g.shape)} != {(n, m)}")

    delta = h_clean.unsqueeze(0) - h_swap        # [R, n, m]  dh_i(j,k), Eq. 3.4
    c = torch.einsum("rnm,nm->rn", delta, g)     # [R, n]     g_i . dh_i(j,k)
    c = c.view(S, K, n)                          # [source j, donor k, carrier i]
    C_js = c.mean(dim=1)                         # [j, i]     Eq. 3.5: average over donors
    return C_js.t().contiguous()                 # [i, j]


def aggregate_carriage_curves(graph_id, distance, F, B, n_boot: int = 2000,
                              boot_seed: int = 1234) -> dict:
    """Distance profiles F(d), B(d), the per-distance MAE mass S(d), and B_far(k).

        F(d)     = mean over {(i,j) : d(i,j) = d} of |C[i,j]|            (Eq. 3.7)
        B(d)     = mean over {(i,j) : d(i,j) = d} of B[i,j]              (analogue of 3.7)
        S(d)     = per-graph SUM of B[i,j] at distance d, averaged over graphs [MAE units]
        B_far(k) = per-graph SUM of B[i,j] over d(i,j) > k, averaged over graphs [MAE units]

    F(d) and B(d) are means over PAIRS, exactly as the definitions state. S(d) and
    B_far(k) are SUMS taken per graph first, which is what keeps them in MAE units and
    makes them telescope: sum_d S(d) over d>k is B_far(k), and the total over all pairs
    is L_clean - L_base for that graph.

    All CIs are 95% bootstrap intervals CLUSTERED ON GRAPHS: pairs inside one molecule
    are strongly dependent, so resampling pairs would understate the interval. For the
    pooled means the bootstrap resamples graphs and recomputes sum(sums)/sum(counts).

    Args:
        graph_id: [P] graph id per pair.
        distance: [P] integer hop distance per pair (finite only; d=inf must be dropped
                  by the caller, since an unreachable pair has no distance bucket).
        F:        [P] |C[i,j]|.
        B:        [P] sign(yhat_clean - y) * C[i,j].
    """
    import numpy as np

    graph_id = np.asarray(graph_id)
    distance = np.asarray(distance).astype(np.int64)
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    dmax = int(distance.max())
    ds = np.arange(0, dmax + 1)
    uniq_g = np.unique(graph_id)
    n_g = int(uniq_g.size)
    gpos = {int(v): p for p, v in enumerate(uniq_g)}
    gidx = np.array([gpos[int(v)] for v in graph_id])

    def _by_graph(vals: np.ndarray, mask: np.ndarray):
        """Per-graph (sum, count). Graphs with no qualifying pair get sum=0, count=0,
        which is the right answer for a SUM (they contribute no error mass)."""
        s = np.zeros(n_g)
        c = np.zeros(n_g)
        np.add.at(s, gidx[mask], vals[mask])
        np.add.at(c, gidx[mask], 1.0)
        return s, c

    def _boot_pooled(s, c):
        if c.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        point = s.sum() / c.sum()
        br = np.random.default_rng(boot_seed)
        draws = br.integers(0, n_g, size=(n_boot, n_g))
        bs, bc = s[draws].sum(axis=1), c[draws].sum(axis=1)
        good = bc > 0
        if not np.any(good):
            return float(point), float("nan"), float("nan")
        vals = bs[good] / bc[good]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(point), float(lo), float(hi)

    def _boot_graph(per_graph):
        br = np.random.default_rng(boot_seed)
        draws = br.integers(0, per_graph.size, size=(n_boot, per_graph.size))
        vals = per_graph[draws].mean(axis=1)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(per_graph.mean()), float(lo), float(hi)

    counts = np.zeros(ds.size, dtype=np.int64)
    F_mean, F_lo, F_hi = (np.full(ds.size, np.nan) for _ in range(3))
    B_mean, B_lo, B_hi = (np.full(ds.size, np.nan) for _ in range(3))
    S_mean, S_lo, S_hi = (np.full(ds.size, np.nan) for _ in range(3))
    F_per_graph = np.full((n_g, ds.size), np.nan)

    for d in ds:
        m = distance == d
        counts[d] = int(m.sum())
        sF, cF = _by_graph(F, m)
        F_mean[d], F_lo[d], F_hi[d] = _boot_pooled(sF, cF)
        with np.errstate(invalid="ignore", divide="ignore"):
            F_per_graph[:, d] = np.where(cF > 0, sF / np.maximum(cF, 1), np.nan)
        sB, cB = _by_graph(B, m)
        B_mean[d], B_lo[d], B_hi[d] = _boot_pooled(sB, cB)
        S_mean[d], S_lo[d], S_hi[d] = _boot_graph(sB)

    ks = np.arange(0, dmax + 1)
    Bf_mean, Bf_lo, Bf_hi = (np.full(ks.size, np.nan) for _ in range(3))
    for k in ks:
        sB, _ = _by_graph(B, distance > k)   # per-graph SUM over d(i,j) > k
        Bf_mean[k], Bf_lo[k], Bf_hi[k] = _boot_graph(sB)

    return {
        "distances": ds, "pair_counts": counts, "n_graphs": n_g,
        "F_mean": F_mean, "F_lo": F_lo, "F_hi": F_hi,
        "B_mean": B_mean, "B_lo": B_lo, "B_hi": B_hi,
        "S_mean": S_mean, "S_lo": S_lo, "S_hi": S_hi,
        "k": ks, "B_far_mean": Bf_mean, "B_far_lo": Bf_lo, "B_far_hi": Bf_hi,
        "F_per_graph": F_per_graph,
    }


# ==================================================================================
# Stage 2: the analysis (runs in-process after prepare_inprocess(); cwd = GRIT repo)
# ==================================================================================

def stage_analyze(args: argparse.Namespace) -> None:
    import json
    import logging

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import grit  # noqa: F401  -- registers GRIT's loaders/encoders/layers/heads
    from torch_geometric import seed_everything
    from torch_geometric.data import Batch
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.comp_budget import params_count

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # GraphGym scratch. Deliberately NOT the training results dir: main.py's
    # custom_set_run_dir() calls makedirs_rm_exist() when auto_resume is False, which
    # would DELETE the trained run. We never call it, and we keep out_dir/run_dir
    # pointed at scratch so nothing in GraphGym can touch the training outputs.
    scratch = out_dir / "_graphgym_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    checks: dict = {}

    # ---------------- config (mirrors main.py, minus the destructive run-dir setup) --
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.work_dir = os.getcwd()
    opts = [
        "out_dir", str(scratch),
        "dataset.dir", str(args.dataset_dir),
        "seed", str(args.seed),
        "accelerator", args.accelerator,
        "wandb.use", "False",
        "mlflow.use", "False",
        "train.auto_resume", "False",
        "train.enable_ckpt", "False",
        "num_threads", str(args.num_threads),
    ]
    # Resolve the config against the GRIT repo so it works no matter the cwd. main.py
    # takes a relative path (cwd=repo); prepare_inprocess() chdirs there, but an absolute
    # path is robust even if a caller invokes stage_analyze directly.
    cfg_file = OFFICIAL_CFG
    if not Path(cfg_file).is_file():
        cand = Path(args.repo_dir) / OFFICIAL_CFG
        if not cand.is_file():
            raise FileNotFoundError(
                f"Official config not found as {OFFICIAL_CFG!r} (cwd={os.getcwd()}) or "
                f"{cand}. Run main() so the GRIT repo is cloned and prepared first, or pass "
                f"--repo-dir pointing at a GRIT checkout."
            )
        cfg_file = str(cand)
    load_cfg(cfg, argparse.Namespace(cfg_file=cfg_file, opts=opts))
    # load_cfg -> assert_cfg sets cfg.run_dir = cfg.out_dir (= scratch). Confirm, because
    # everything downstream that GraphGym might write must stay away from the trained run.
    assert Path(cfg.run_dir).resolve() == scratch.resolve(), \
        f"cfg.run_dir unexpectedly {cfg.run_dir}; refusing to run near training outputs."

    # cfg.device must be valid BEFORE create_model(): create_model(to_device=True) moves
    # the model to cfg.device, so a stale 'cuda:0' on a CPU runtime would fail there.
    device = torch.device(args.accelerator if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)
    if device.type != "cuda":
        log("[warn] CUDA unavailable; running on CPU. This will be slow.")
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)

    # ---------------- data + model -------------------------------------------------
    log("\n[data] Building loaders (RRWP is recomputed in-memory for all 12k ZINC graphs; "
        "GRIT's pre_transform_in_memory does not cache to disk, so expect a few minutes).")
    loaders = create_loader()  # [train, val, test]
    assert len(loaders) == 3, f"Expected [train, val, test] loaders, got {len(loaders)}"
    split_of = {"train": 0, "val": 1, "test": 2}
    eval_ds = loaders[split_of[args.eval_split]].dataset
    donor_ds = loaders[split_of[args.donor_split]].dataset
    log(f"[data] eval split={args.eval_split} ({len(eval_ds)} graphs) | "
        f"donor split={args.donor_split} ({len(donor_ds)} graphs)")

    model = create_model()
    n_params = params_count(model)
    log(f"[model] Num parameters: {n_params}")
    checks["num_parameters"] = int(n_params)
    if n_params != EXPECTED_ZINC_GRIT_RRWP_PARAMS:
        msg = (f"[param-check:ERROR] parameter count {n_params} != "
               f"{EXPECTED_ZINC_GRIT_RRWP_PARAMS} (GRIT paper ZINC row).")
        if not args.allow_param_count_drift:
            raise RuntimeError(msg)
        log(msg + " Continuing due to --allow-param-count-drift.")
    else:
        log(f"[param-check] Matched expected GRIT ZINC+RRWP parameter count: {n_params}")

    # ---------------- checkpoint ---------------------------------------------------
    ckpt_path = Path(args.ckpt)
    log(f"\n[ckpt] Loading: {ckpt_path}")
    blob = torch.load(str(ckpt_path), map_location="cpu")
    # GraphGym's save_ckpt() stores ONLY {model_state, optimizer_state, scheduler_state};
    # the epoch is carried by the filename (get_ckpt_path -> f'{epoch}.ckpt'), not by a
    # key inside the blob. So the filename is authoritative here.
    ckpt_epoch = int(ckpt_path.stem) if ckpt_path.stem.isdigit() else -1
    if isinstance(blob, dict) and "model_state" in blob:
        state = blob["model_state"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    else:
        state = blob
    if isinstance(blob, dict) and isinstance(blob.get("epoch"), int):
        ckpt_epoch = int(blob["epoch"])
    log(f"[ckpt] epoch (from filename): {ckpt_epoch}")

    def _try_load(sd) -> tuple[bool, str]:
        try:
            model.load_state_dict(sd, strict=True)
            return True, "strict"
        except RuntimeError as e:
            return False, str(e)

    ok, how = _try_load(state)
    if not ok:
        # save_ckpt stores GraphGymModule.state_dict(), so keys should already carry the
        # "model." prefix. Handle both directions rather than silently loading nothing.
        stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        prefixed = {f"model.{k}": v for k, v in state.items()}
        for cand, label in ((stripped, "stripped 'model.' prefix"), (prefixed, "added 'model.' prefix")):
            if cand:
                ok, how2 = _try_load(cand)
                if ok:
                    how = label
                    break
        if not ok:
            raise RuntimeError(f"Could not load checkpoint state_dict strictly.\n{how}")
    log(f"[ckpt] Loaded strictly (key handling: {how}); 0 missing / 0 unexpected keys.")
    checks["ckpt_path"] = str(ckpt_path)
    checks["ckpt_epoch"] = int(ckpt_epoch) if isinstance(ckpt_epoch, int) else -1
    checks["ckpt_key_handling"] = how

    model.to(device)
    # eval() is load-bearing, not hygiene: gt.batch_norm=True and gt.attn_dropout=0.2
    # would otherwise make a replica's output depend on the rest of the batch, which
    # would silently corrupt every swap evaluated in a shared forward pass.
    model.eval()

    assert type(model.model).__name__ == "GritTransformer", \
        f"Expected GritTransformer, got {type(model.model).__name__}"

    # ---------------- check 4: recompute test MAE from the loaded checkpoint --------
    @torch.no_grad()
    def split_mae(loader) -> float:
        tot, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            pred, true = model(batch)
            p, t = pred.view(-1), true.view(-1)
            assert p.numel() == t.numel()
            tot += (p - t).abs().sum().item()
            cnt += t.numel()
        return tot / max(cnt, 1)

    if args.eval_mae:
        t0 = time.perf_counter()
        test_mae = split_mae(loaders[2])
        log(f"\n[verify] Test MAE recomputed from checkpoint: {test_mae:.5f} "
            f"(GRIT paper ZINC-subset ~{PAPER_ZINC_TEST_MAE}) [{time.perf_counter()-t0:.1f}s]")
        checks["test_mae"] = float(test_mae)
        if test_mae > args.mae_sanity_threshold:
            raise RuntimeError(
                f"Test MAE {test_mae:.5f} exceeds --mae-sanity-threshold "
                f"{args.mae_sanity_threshold}. The checkpoint almost certainly did not "
                f"load correctly, or training has barely progressed. Refusing to report "
                f"carriage for a model that does not reproduce its own test metric."
            )
    else:
        checks["test_mae"] = None

    # ---------------- h^L capture hook ---------------------------------------------
    # GritTransformer.forward walks self.children(): encoder -> rrwp_abs_encoder ->
    # rrwp_rel_encoder -> layers (10x GritTransformerLayer) -> post_mp (SANGraphHead).
    # The tensor entering post_mp is batch.x out of `layers`: that is h^L.
    store: dict = {}

    def _hook(_module, _inputs, output):
        store["h"] = output.x

    handle = model.model.layers.register_forward_hook(_hook)

    num_types = int(cfg.dataset.node_encoder_num_types)
    dim_h = int(cfg.gnn.dim_inner)

    # ---------------- donor pool (Def 3.2.2) ---------------------------------------
    log(f"\n[donors] Building donor pool from the '{args.donor_split}' split "
        f"(real node content = on-manifold; Def 3.2.2).")
    donor_types_all: List[int] = []
    donor_gid_all: List[int] = []
    for gi in range(len(donor_ds)):
        d = donor_ds[gi]
        xs = d.x[:, 0].tolist()
        donor_types_all.extend(int(v) for v in xs)
        donor_gid_all.extend([gi] * len(xs))
    donor_types_all = np.asarray(donor_types_all, dtype=np.int64)
    donor_gid_all = np.asarray(donor_gid_all, dtype=np.int64)
    assert donor_types_all.min() >= 0 and donor_types_all.max() < num_types, (
        f"Donor atom types out of range for node_encoder_num_types={num_types}: "
        f"[{donor_types_all.min()}, {donor_types_all.max()}]"
    )
    log(f"[donors] pool size = {donor_types_all.size} nodes over {len(donor_ds)} graphs; "
        f"{len(np.unique(donor_types_all))} distinct atom types (max type index "
        f"{donor_types_all.max()} < {num_types}).")

    # ---------------- graph selection ----------------------------------------------
    rng = np.random.default_rng(args.analysis_seed)
    n_graphs = min(args.num_graphs, len(eval_ds))
    if args.graph_select == "random":
        graph_ids = np.sort(rng.choice(len(eval_ds), size=n_graphs, replace=False))
    else:
        graph_ids = np.arange(n_graphs)
    log(f"[select] {n_graphs} eval graphs ({args.graph_select}, seed={args.analysis_seed}), "
        f"K={args.donors} donors per source.")

    K = int(args.donors)

    # ---------------- per-pair accumulators ----------------------------------------
    all_gid: List[np.ndarray] = []
    all_i: List[np.ndarray] = []
    all_j: List[np.ndarray] = []
    all_d: List[np.ndarray] = []
    all_C: List[np.ndarray] = []
    all_B: List[np.ndarray] = []
    # additivity diagnostic, per (graph, source)
    add_sumC: List[np.ndarray] = []
    add_dyhat: List[np.ndarray] = []

    g_spread_max = 0.0
    noop_total = 0
    noop_max_dh = 0.0
    donor_draws = 0
    unreachable_total = 0
    batchinv_max = 0.0
    struct_checked = 0
    peak_mem = 0
    bexact_max = 0.0   # max |sum_i B[i,j] - dL_j| over sources that moved the output

    def _spd(data, n: int) -> np.ndarray:
        ei = data.edge_index.cpu().numpy()
        if ei.shape[1] == 0:
            D = np.full((n, n), np.inf)
            np.fill_diagonal(D, 0.0)
            return D
        A = csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
        return shortest_path(A, method="D", unweighted=True, directed=False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    for gi_pos, gi in enumerate(graph_ids):
        base = eval_ds[int(gi)]
        n = int(base.num_nodes)
        y = float(base.y.view(-1)[0].item())

        # --- distances from the PRISTINE graph. This must happen before any forward
        # --- pass: GRIT's rrwp_rel_encoder overwrites batch.edge_index with the full
        # --- (all-pairs) index, which would make every distance 1.
        D = _spd(base, n)
        n_unreach = int(np.isinf(D).sum())
        unreachable_total += n_unreach

        # --- clean pass: h^L and g_i = d yhat / d h^L_i  (Eq 3.1) --------------------
        cb = Batch.from_data_list([base]).to(device)
        with torch.enable_grad():
            pred_c, _ = model(cb)
            h_clean_t = store["h"]
            assert h_clean_t.shape == (n, dim_h), f"h^L shape {tuple(h_clean_t.shape)} != {(n, dim_h)}"
            g_t = torch.autograd.grad(pred_c.sum(), h_clean_t)[0]  # [n, dim_h]
        yhat_clean = float(pred_c.view(-1)[0].item())
        h_clean = h_clean_t.detach()
        g = g_t.detach()

        # check 5: sum pooling => g_i identical for all i.
        g_spread_max = max(g_spread_max, float((g - g[0:1]).abs().max().item()))

        # --- donors: K per source j, independently sampled (D_j is indexed by j) ----
        if args.donor_split == args.eval_split:
            # Def 3.2.2: the donor comes from ANOTHER graph G' != G.
            pool = np.flatnonzero(donor_gid_all != int(gi))
        else:
            # Different split => no graph overlap with G by construction.
            pool = np.arange(donor_types_all.size)
        donor_ids = rng.choice(pool, size=(n, K), replace=True)
        donor_type = donor_types_all[donor_ids]                 # [n, K]
        own_type = base.x[:, 0].cpu().numpy()                   # [n]
        noop_mask = donor_type == own_type[:, None]             # [n, K]
        noop_total += int(noop_mask.sum())
        donor_draws += donor_type.size

        # --- swap forwards ----------------------------------------------------------
        S = n                     # one source per node
        R = S * K                 # replicas
        # Replica r = j*K + k, matching donor_type.reshape(-1) and the [S, K, n] view
        # taken in carriage_from_states. Batch.from_data_list stacks graphs contiguously,
        # so node u of replica r sits at row r*n + u.
        rows_local = np.repeat(np.arange(S), K)   # the perturbed node inside each replica
        flat_donor = donor_type.reshape(-1)       # [R]

        h_swap = torch.empty((R, n, dim_h), device=device, dtype=h_clean.dtype)
        yhat_swap = torch.empty((R,), device=device, dtype=torch.float32)

        chunk = _plan_chunk(n, args)
        r0 = 0
        while r0 < R:
            m = min(chunk, R - r0)
            while True:
                try:
                    with torch.no_grad():
                        b = Batch.from_data_list([base] * m).to(device)
                        # Batch collation cats tensors, so b.x is fresh storage and the
                        # dataset's cached tensors are never touched.
                        rows = (torch.arange(m, device=device) * n
                                + torch.as_tensor(rows_local[r0:r0 + m], device=device))
                        b.x[rows, 0] = torch.as_tensor(
                            flat_donor[r0:r0 + m], device=device, dtype=b.x.dtype)

                        if args.verify and struct_checked < args.verify_graphs and r0 == 0:
                            _verify_structure(base, b, m, n, rows, log)
                            struct_checked += 1

                        pred_s, _ = model(b)
                        hs = store["h"]
                        assert hs.shape == (m * n, dim_h)
                        h_swap[r0:r0 + m] = hs.view(m, n, dim_h)
                        yhat_swap[r0:r0 + m] = pred_s.view(-1).float()
                    break
                except RuntimeError as exc:
                    # torch.cuda.OutOfMemoryError subclasses RuntimeError and only exists
                    # from torch 1.13; matching the message keeps this portable.
                    if "out of memory" not in str(exc).lower():
                        raise
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if m == 1:
                        raise
                    m = max(1, m // 2)
                    chunk = m
                    log(f"[mem] CUDA OOM -> reducing replicas/forward to {m}")
            r0 += m

        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

        # --- check 6: batch invariance (clean graph alone vs inside a big batch) ----
        if args.verify and gi_pos < args.verify_graphs:
            with torch.no_grad():
                mm = min(_plan_chunk(n, args), 64)
                b = Batch.from_data_list([base] * mm).to(device)
                model(b)
                hb = store["h"].view(mm, n, dim_h)
                dev = (hb - h_clean.unsqueeze(0)).abs().max().item()
            batchinv_max = max(batchinv_max, float(dev))

        # --- check 7: no-op donors must give exactly dh = 0 -------------------------
        if noop_mask.any():
            nm = torch.as_tensor(noop_mask.reshape(-1), device=device)
            dh_noop = (h_swap[nm] - h_clean.unsqueeze(0)).abs().max().item()
            noop_max_dh = max(noop_max_dh, float(dh_noop))

        # --- carriage: Eq 3.4 / 3.5 -------------------------------------------------
        C = carriage_from_states(h_clean, h_swap, g, S, K).cpu().numpy()   # C[i, j]

        # --- beneficial carriage ----------------------------------------------------
        # EXACT per-source loss change, attributed to carriers by their share of the
        # output change. This replaces the linearized B = sign(yhat_clean - y) * C
        # (Eq. 3.8) with a form that is exact across the |.| kink and uses the ACTUAL
        # post-swap loss (so it sees that corrupting a node's own content is catastrophic,
        # which the linearization -- read off the clean point -- cannot).
        #
        #   dL_j        = |yhat_clean - y| - mean_k |yhat_swap(j,k) - y|      [per source]
        #   B[i,j]      = dL_j * ( C[i,j] / sum_i C[i,j] )                    [carrier share]
        #   => sum_i B[i,j] = dL_j  EXACTLY;  B<0 = beneficial, B>0 = adverse (MAE units)
        #
        # Donor-average the LOSS, not the prediction: the kink makes E|.| != |E.| exactly
        # when donor variance is large, so averaging yhat over donors and then taking |.|
        # would misstate benefit near a residual sign crossing.
        L_clean = abs(yhat_clean - y)
        L_swap = (yhat_swap.view(S, K) - y).abs().mean(dim=1).cpu().numpy()   # [S] per source
        dL_j = L_clean - L_swap                                               # [S]  <0 = beneficial
        sumC_j = C.sum(axis=0)                                                # [n]  sum_i C[i,j]
        eps = 1e-9
        # Guard the denominator: a source that moves nothing (sum_i C[i,j] ~ 0) also has
        # dL_j ~ 0, so zeroing its column is the correct, well-defined attribution.
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(np.abs(sumC_j) > eps, C / sumC_j, 0.0)          # C[i,j] / sum_i C[i,j]
        B = dL_j[None, :] * share                                            # sum_i B[i,j] = dL_j

        # --- additivity diagnostic (check 9): audits sum_i C[i,j] ~ yhat_clean-yhat_base,
        # --- i.e. that the carrier "share" divides a meaningful output change. -------
        dyhat_j = yhat_clean - yhat_swap.view(S, K).mean(dim=1).cpu().numpy()
        add_sumC.append(sumC_j)
        add_dyhat.append(dyhat_j)
        # source-level exactness of the new B (must hold to fp tolerance where sumC!=0)
        moved = np.abs(sumC_j) > eps
        if moved.any():
            bexact_max = max(bexact_max,
                             float(np.abs(B.sum(axis=0)[moved] - dL_j[moved]).max()))

        # --- flatten pairs ----------------------------------------------------------
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        finite = np.isfinite(D)
        all_gid.append(np.full(int(finite.sum()), int(gi), dtype=np.int64))
        all_i.append(ii[finite].astype(np.int64))
        all_j.append(jj[finite].astype(np.int64))
        all_d.append(D[finite].astype(np.int64))
        all_C.append(C[finite].astype(np.float64))
        all_B.append(B[finite].astype(np.float64))

        del h_swap, yhat_swap
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (gi_pos + 1) % max(1, n_graphs // 10) == 0 or gi_pos == 0:
            log(f"[run] graph {gi_pos+1}/{n_graphs} (id={int(gi)}, n={n}, "
                f"{R} swap forwards) | {time.perf_counter()-t_start:.1f}s")

    handle.remove()

    gid = np.concatenate(all_gid)
    pi = np.concatenate(all_i)
    pj = np.concatenate(all_j)
    pd = np.concatenate(all_d)
    pC = np.concatenate(all_C)
    pB = np.concatenate(all_B)
    F = np.abs(pC)                                                 # (3.6)

    # ---------------- report the checks ---------------------------------------------
    log("\n" + "=" * 84)
    log("VERIFICATION")
    log("=" * 84)
    log(f"  [1] parameters                 : {n_params} (expected {EXPECTED_ZINC_GRIT_RRWP_PARAMS})")
    log(f"  [3] checkpoint                 : {ckpt_path.name} (epoch {checks['ckpt_epoch']}), "
        f"strict load OK")
    if checks["test_mae"] is not None:
        log(f"  [4] test MAE from checkpoint   : {checks['test_mae']:.5f} "
            f"(paper ~{PAPER_ZINC_TEST_MAE})")
    log(f"  [5] max_i |g_i - g_1|          : {g_spread_max:.3e}  "
        f"(sum pooling => must be ~0)")
    if args.verify:
        log(f"  [6] batch-invariance max|dh|   : {batchinv_max:.3e}  "
            f"(clean alone vs in a replica batch)")
        log(f"  [8] structure invariance       : verified on {struct_checked} graph(s): only "
            f"x changed; edge_index/edge_attr/rrwp* bit-identical")
    noop_frac = noop_total / max(donor_draws, 1)
    log(f"  [7] no-op donors               : {noop_total}/{donor_draws} "
        f"({100*noop_frac:.1f}%) with max|dh| = {noop_max_dh:.3e} (must be exactly 0)")
    checks["g_spread_max"] = float(g_spread_max)
    checks["batch_invariance_max_abs_dh"] = float(batchinv_max)
    checks["donor_noop_fraction"] = float(noop_frac)
    checks["donor_noop_max_abs_dh"] = float(noop_max_dh)
    checks["structure_verified_graphs"] = int(struct_checked)

    tol = args.tol
    if g_spread_max > tol:
        log(f"  [!] g_i varies across nodes by {g_spread_max:.3e} > tol {tol:.1e}. The readout "
            f"is not behaving as sum-pool -> MLP; interpret g_i with care.")
    if noop_max_dh > tol:
        raise RuntimeError(
            f"No-op donors produced |dh| = {noop_max_dh:.3e} > tol {tol:.1e}. A swap that "
            f"replaces an atom type with the SAME atom type must be a bit-exact identity. "
            f"Something other than node j's content is changing between passes "
            f"(model not in eval(), or the wrong row is being edited)."
        )
    if args.verify and batchinv_max > tol:
        raise RuntimeError(
            f"Batch invariance violated: max|dh| = {batchinv_max:.3e} > tol {tol:.1e}. "
            f"Evaluating replicas in a shared forward pass is not safe here."
        )
    if unreachable_total:
        log(f"  [10] unreachable pairs        : {unreachable_total} pairs with d=inf were "
            f"EXCLUDED from all distance statistics (ZINC molecules are normally connected).")
    else:
        log("  [10] unreachable pairs        : 0 (all evaluated molecules connected)")
    checks["unreachable_pairs_excluded"] = int(unreachable_total)

    a_sumC = np.concatenate(add_sumC)
    a_dyhat = np.concatenate(add_dyhat)
    if a_sumC.size > 2 and np.std(a_sumC) > 0 and np.std(a_dyhat) > 0:
        r = float(np.corrcoef(a_sumC, a_dyhat)[0, 1])
        slope = float(np.polyfit(a_sumC, a_dyhat, 1)[0])
    else:
        r, slope = float("nan"), float("nan")
    log(f"  [9] additivity sum_i C[i,j] vs (yhat_clean - mean_k yhat_swap): "
        f"r={r:.4f}, slope={slope:.4f}")
    log("      (audits that the carrier 'share' C[i,j]/sum_i C[i,j] divides a meaningful "
        "output change; r<1 is expected -- the head MLP after sum-pooling is nonlinear)")
    checks["additivity_pearson_r"] = r
    checks["additivity_slope"] = slope
    log(f"  [11] beneficial exactness       : max_j |sum_i B[i,j] - dL_j| = {bexact_max:.3e}  "
        f"(B decomposes the exact per-source loss change; must be ~0)")
    checks["beneficial_exactness_max"] = float(bexact_max)
    if bexact_max > tol:
        raise RuntimeError(
            f"Beneficial-carriage exactness violated: max_j |sum_i B[i,j] - dL_j| = "
            f"{bexact_max:.3e} > tol {tol:.1e}. B is defined so its column sum equals the "
            f"exact per-source loss change dL_j; a mismatch means the share attribution "
            f"or the loss donor-average is miswired."
        )
    if device.type == "cuda":
        log(f"  [mem] peak CUDA memory allocated: {peak_mem/1024**3:.2f} GiB of "
            f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")
        checks["peak_cuda_gib"] = float(peak_mem / 1024**3)

    # ---------------- aggregation ---------------------------------------------------
    agg = aggregate_carriage_curves(gid, pd, F, pB, n_boot=args.n_boot, boot_seed=args.boot_seed)
    ds = agg["distances"]
    counts = agg["pair_counts"]
    F_mean, F_lo, F_hi = agg["F_mean"], agg["F_lo"], agg["F_hi"]
    B_mean, B_lo, B_hi = agg["B_mean"], agg["B_lo"], agg["B_hi"]
    S_mean, S_lo, S_hi = agg["S_mean"], agg["S_lo"], agg["S_hi"]
    ks = agg["k"]
    Bf_mean, Bf_lo, Bf_hi = agg["B_far_mean"], agg["B_far_lo"], agg["B_far_hi"]
    F_per_graph_curves = agg["F_per_graph"]
    n_g = agg["n_graphs"]

    # ---------------- figures --------------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    C_FUNC, C_BEN, C_ADV = "#2b6cb0", "#2f855a", "#c53030"
    tag = (f"GRIT+RRWP ZINC-subset | epoch {checks['ckpt_epoch']}"
           + (f" | test MAE {checks['test_mae']:.4f}" if checks["test_mae"] is not None else "")
           + f"\n{n_g} {args.eval_split} graphs, K={K} donor swaps/source, "
             f"donors from '{args.donor_split}'")

    # ---- Fig 1: F(d) and B(d)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.fill_between(ds, F_lo, F_hi, color=C_FUNC, alpha=0.18, lw=0)
    ax.plot(ds, F_mean, "o-", color=C_FUNC, lw=1.8, ms=4.5)
    ax.set_xlabel("shortest-path distance $d(i,j)$ [hops]")
    ax.set_ylabel(r"$F(d)=\mathrm{mean}_{d(i,j)=d}\,|C[i,j]|$")
    ax.set_title("Functional carriage $F(d)$\n(label-free: does the model use $j$ at $i$?)")
    if np.nanmin(F_mean) > 0:  # log axis is the informative one, but only if it is legal
        ax.set_yscale("log")
    ax.set_xticks(ds)

    ax = axes[1]
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ds, B_lo, B_hi, color="0.5", alpha=0.18, lw=0)
    ax.plot(ds, B_mean, "o-", color="0.15", lw=1.8, ms=4.5, zorder=3)
    ax.fill_between(ds, np.minimum(B_mean, 0), 0, color=C_BEN, alpha=0.30, lw=0)
    ax.fill_between(ds, np.maximum(B_mean, 0), 0, color=C_ADV, alpha=0.30, lw=0)
    ax.set_xlabel("shortest-path distance $d(i,j)$ [hops]")
    ax.set_ylabel(r"$B(d)=\mathrm{mean}_{d(i,j)=d}\,B[i,j]$   [MAE units]")
    ax.set_title("Beneficial carriage $B(d)$\n"
                 r"$B<0$ beneficial $\cdot$ $B>0$ adverse $\cdot$ $B\approx0$ dispensable")
    ax.set_xticks(ds)
    ax.text(0.985, 0.05, "beneficial", transform=ax.transAxes, ha="right", va="bottom",
            color=C_BEN, fontsize=8.5, fontweight="bold")
    ax.text(0.985, 0.95, "adverse", transform=ax.transAxes, ha="right", va="top",
            color=C_ADV, fontsize=8.5, fontweight="bold")

    fig.suptitle("Semantic donor-swap interventions on GRIT/ZINC   " + tag, fontsize=9.5, y=1.10)
    p1 = fig_dir / "fig_semantic_carriage_Fd_Bd.png"
    fig.savefig(p1); plt.close(fig)
    log(f"\n[fig] {p1}")

    # ---- Fig 2: B_far(k)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ks, Bf_lo, Bf_hi, color="0.5", alpha=0.20, lw=0)
    ax.plot(ks, Bf_mean, "o-", color="0.15", lw=2.0, ms=5, zorder=3)
    ax.fill_between(ks, np.minimum(Bf_mean, 0), 0, color=C_BEN, alpha=0.30, lw=0)
    ax.fill_between(ks, np.maximum(Bf_mean, 0), 0, color=C_ADV, alpha=0.30, lw=0)
    ax.set_xlabel("hop threshold $k$")
    ax.set_ylabel(r"$B_{\mathrm{far}}(k)=\sum_{\{(i,j):\,d(i,j)>k\}} B[i,j]$   [MAE units]")
    ax.set_title(r"Beneficial carriage beyond $k$ hops"
                 "\n" r"per graph, mean over graphs (95% CI, bootstrap over graphs)")
    ax.set_xticks(ks)
    ax.text(0.985, 0.05, "net beneficial", transform=ax.transAxes, ha="right", va="bottom",
            color=C_BEN, fontsize=9, fontweight="bold")
    ax.text(0.985, 0.95, "net adverse", transform=ax.transAxes, ha="right", va="top",
            color=C_ADV, fontsize=9, fontweight="bold")
    fig.suptitle(tag, fontsize=8.5, y=1.02)
    p2 = fig_dir / "fig_semantic_carriage_Bfar.png"
    fig.savefig(p2); plt.close(fig)
    log(f"[fig] {p2}")

    # ---- Fig 3: diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax = axes[0, 0]
    ax.bar(ds, counts, color=C_FUNC, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("$d(i,j)$ [hops]"); ax.set_ylabel("# pairs")
    ax.set_title("Pair support per distance")
    ax.set_xticks(ds)

    ax = axes[0, 1]
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ds, S_lo, S_hi, color="0.5", alpha=0.18, lw=0)
    ax.plot(ds, S_mean, "o-", color="0.15", lw=1.8, ms=4.5)
    ax.set_xlabel("$d(i,j)$ [hops]")
    ax.set_ylabel(r"$\sum_{d(i,j)=d} B[i,j]$ per graph  [MAE units]")
    ax.set_title("Error mass carried at each distance\n"
                 r"(tail sums give $B_{\mathrm{far}}(k)$; per source $\sum_i B[i,j]=dL_j$ exactly)")
    ax.set_xticks(ds)

    ax = axes[1, 0]
    ax.scatter(a_sumC, a_dyhat, s=6, alpha=0.28, color=C_FUNC, edgecolors="none")
    lim = max(float(np.nanmax(np.abs(np.concatenate([a_sumC, a_dyhat])))) * 1.05, 1e-12)
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1.0, label="$y=x$")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\sum_i C[i,j]$")
    ax.set_ylabel(r"$\hat{y}_{\mathrm{clean}} - \mathrm{mean}_k\,\hat{y}_{\mathrm{swap}}(j,k)$")
    ax.set_title(f"Additivity audit (first order)\n$r$={r:.3f}, slope={slope:.3f}")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for gi_ in range(n_g):
        ax.plot(ds, F_per_graph_curves[gi_], "-", color=C_FUNC, alpha=0.13, lw=0.9)
    ax.plot(ds, F_mean, "o-", color="0.1", lw=2.0, ms=4.5, label="pooled $F(d)$")
    if np.nanmin(F_mean) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("$d(i,j)$ [hops]"); ax.set_ylabel("$F(d)$")
    ax.set_title("Per-graph $F(d)$ (spaghetti) vs pooled")
    ax.set_xticks(ds)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Diagnostics   " + tag, fontsize=9.5, y=1.03)
    p3 = fig_dir / "fig_semantic_carriage_diagnostics.png"
    fig.savefig(p3); plt.close(fig)
    log(f"[fig] {p3}")

    # ---------------- persist --------------------------------------------------------
    npz_path = out_dir / "carriage_pairs.npz"
    np.savez_compressed(
        npz_path,
        graph_id=gid, carrier_i=pi, source_j=pj, distance=pd,
        C=pC, B=pB, F=F,
        distances=ds, F_mean=F_mean, F_lo=F_lo, F_hi=F_hi,
        B_mean=B_mean, B_lo=B_lo, B_hi=B_hi,
        B_sum_per_graph_mean=S_mean, B_sum_per_graph_lo=S_lo, B_sum_per_graph_hi=S_hi,
        pair_counts=counts, k=ks, B_far_mean=Bf_mean, B_far_lo=Bf_lo, B_far_hi=Bf_hi,
        additivity_sumC=a_sumC, additivity_dyhat=a_dyhat,
    )
    log(f"[data] {npz_path}")

    summary = {
        "provenance": {
            "grit_repo": OFFICIAL_REPO, "grit_commit": OFFICIAL_COMMIT,
            "config": OFFICIAL_CFG, "checkpoint": str(ckpt_path),
            "checkpoint_epoch": checks["ckpt_epoch"],
        },
        "settings": {
            "eval_split": args.eval_split, "donor_split": args.donor_split,
            "num_graphs": int(n_g), "donors_K": K,
            "graph_select": args.graph_select, "analysis_seed": args.analysis_seed,
            "n_boot": args.n_boot,
        },
        "checks": checks,
        "curves": {
            "distance": ds.tolist(),
            "pair_counts": counts.tolist(),
            "F_mean": F_mean.tolist(), "F_lo": F_lo.tolist(), "F_hi": F_hi.tolist(),
            "B_mean": B_mean.tolist(), "B_lo": B_lo.tolist(), "B_hi": B_hi.tolist(),
            "B_sum_per_graph_mean": S_mean.tolist(),
            "k": ks.tolist(),
            "B_far_mean": Bf_mean.tolist(), "B_far_lo": Bf_lo.tolist(), "B_far_hi": Bf_hi.tolist(),
        },
        "figures": [str(p1), str(p2), str(p3)],
    }
    json_path = out_dir / "carriage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[data] {json_path}")

    # ---------------- console table --------------------------------------------------
    log("\n" + "=" * 84)
    log("RESULTS  (C = clean - swapped, donor-averaged; B<0 = beneficial, MAE units)")
    log("=" * 84)
    log(f"{'d':>3} {'#pairs':>8} {'F(d)':>12} {'B(d)':>13} {'sum_d B/graph':>15}")
    for d in ds:
        log(f"{d:>3} {counts[d]:>8} {F_mean[d]:>12.4e} {B_mean[d]:>13.3e} {S_mean[d]:>15.3e}")
    log("")
    log(f"{'k':>3} {'B_far(k) [MAE]':>16} {'95% CI':>28}")
    for k in ks:
        log(f"{k:>3} {Bf_mean[k]:>16.4e}   [{Bf_lo[k]:>+.3e}, {Bf_hi[k]:>+.3e}]")
    log("\n[done] Semantic-intervention carriage analysis complete.")


def _plan_chunk(n: int, args: argparse.Namespace) -> int:
    """Replicas per forward pass.

    GRIT pads to full attention, so a replica of an n-node graph costs ~n^2 pair-edges.
    We budget on both pair-edges and replica count and let an OOM backoff halve if the
    estimate is optimistic. On an A100 (40 or 80 GB) with ZINC's ~23-node molecules the
    defaults put every source x donor replica of a graph into a single forward pass; peak
    memory is only a few GB, so 40 GB is ample and the OOM backoff is just a safety net.
    """
    per = max(n * n, 1)
    return max(1, min(args.max_replicas, args.max_pair_edges // per))


def _verify_structure(base, b, m: int, n: int, rows, log_fn) -> None:
    """Check 8: the intervention changed content and nothing else.

    Runs on the collated batch BEFORE the forward pass mutates it (GRIT's encoders
    rebind batch.x / batch.edge_index / batch.edge_attr in place of the raw inputs).
    """
    import torch

    def _rep(t: "torch.Tensor", inc: int = 0) -> "torch.Tensor":
        """Replicate a per-graph tensor m times the way Batch collation would."""
        parts = []
        for r in range(m):
            parts.append(t + (r * n if inc else 0))
        return torch.cat(parts, dim=(-1 if inc else 0))

    ei = _rep(base.edge_index.to(b.edge_index.device), inc=1)
    assert torch.equal(b.edge_index, ei), "edge_index changed under a semantic intervention"
    assert torch.equal(b.edge_attr, _rep(base.edge_attr.to(b.edge_attr.device))), \
        "edge_attr changed under a semantic intervention"
    assert torch.equal(b.rrwp, _rep(base.rrwp.to(b.rrwp.device))), \
        "rrwp (node PE) changed under a semantic intervention"
    assert torch.equal(b.rrwp_val, _rep(base.rrwp_val.to(b.rrwp_val.device))), \
        "rrwp_val (pair PE) changed under a semantic intervention"
    assert torch.equal(b.rrwp_index, _rep(base.rrwp_index.to(b.rrwp_index.device), inc=1)), \
        "rrwp_index changed under a semantic intervention"

    x_ref = _rep(base.x.to(b.x.device))
    diff = (b.x != x_ref).any(dim=1)
    assert int(diff.sum().item()) <= m, "more than one node per replica was modified"
    changed = set(diff.nonzero().view(-1).tolist())
    assert changed.issubset(set(rows.tolist())), \
        "a node other than the intended source j was modified"
    log_fn("[verify] structure invariance OK: edge_index/edge_attr/rrwp/rrwp_val/rrwp_index "
           "bit-identical; x differs only at the intended source node(s).")


# ==================================================================================
# CLI
# ==================================================================================

def _strip_colab_kernel_args(argv: Sequence[str]) -> List[str]:
    """Drop the `-f /root/.../kernel-xxx.json` argv IPython injects, keep real typos strict."""
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if (a == "-f" and i + 1 < len(argv)
                and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json")):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a} {argv[i+1]}")
            i += 2
            continue
        if a.startswith("-f=") and "kernel-" in a and a.endswith(".json"):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a}")
            i += 1
            continue
        cleaned.append(a)
        i += 1
    return cleaned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Semantic-intervention carriage F(d)/B(d)/B_far(k) for a GRIT ZINC checkpoint.",
        epilog=textwrap.dedent(
            """
            Examples:
              from grit_zinc_semantic_carriage_colab import main
              main(["--skip-install"])
              main(["--skip-install", "--num-graphs", "128", "--donors", "64"])
              main(["--skip-install", "--ckpt", "/content/drive/MyDrive/.../ckpt/1873.ckpt"])
            """
        ),
    )
    # environment
    p.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    p.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/grit_zinc_official"),
                   help="Same --drive-dir the training runner used.")
    p.add_argument("--repo-dir", type=Path, default=Path("/content/GRIT"))
    p.add_argument("--repo-url", type=str, default=OFFICIAL_REPO)
    p.add_argument("--branch", type=str, default="main")
    p.add_argument("--commit", type=str, default=OFFICIAL_COMMIT,
                   help="Pin GRIT to this commit. Empty string = branch HEAD.")
    p.add_argument("--pyg-version", type=str, default="2.2.0")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--force-fresh-repo", action="store_true")
    p.add_argument("--allow-upstream-config-drift", action="store_true")
    p.add_argument("--allow-param-count-drift", action="store_true")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--accelerator", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42, help="cfg.seed; match the training run.")

    # checkpoint
    p.add_argument("--ckpt", type=str, default=None,
                   help="Explicit checkpoint path. Default: auto-discover the last one on Drive.")

    # analysis
    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: <drive-dir>/carriage_analysis")
    p.add_argument("--dataset-dir", type=str, default=None,
                   help="Default: <drive-dir>/datasets (reuses the training run's ZINC download).")
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--donor-split", choices=["train", "val", "test"], default="test",
                   help="Def 3.2.2 draws donors from 'the dataset from which G is drawn', so this "
                        "defaults to the eval split. 'train' is statistically equivalent (same "
                        "atom marginal) and is offered for robustness checks.")
    p.add_argument("--num-graphs", type=int, default=64)
    p.add_argument("--donors", type=int, default=32, help="K in Eq. 3.5.")
    p.add_argument("--graph-select", choices=["random", "first"], default="random")
    p.add_argument("--analysis-seed", type=int, default=0,
                   help="Seeds graph selection and donor sampling.")
    p.add_argument("--boot-seed", type=int, default=1234)
    p.add_argument("--n-boot", type=int, default=2000)

    # verification
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no-verify", action="store_false", dest="verify")
    p.add_argument("--verify-graphs", type=int, default=2,
                   help="How many graphs get the costly batch-invariance/structure checks.")
    p.add_argument("--eval-mae", action="store_true", default=True,
                   help="Recompute test MAE from the checkpoint (the strongest load check).")
    p.add_argument("--no-eval-mae", action="store_false", dest="eval_mae")
    p.add_argument("--mae-sanity-threshold", type=float, default=0.15,
                   help="Abort if recomputed test MAE exceeds this (paper is ~0.059).")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Tolerance for the exactness checks (fp32 GRIT forward).")

    # throughput
    p.add_argument("--max-replicas", type=int, default=4096,
                   help="Max perturbed replicas per forward pass.")
    p.add_argument("--max-pair-edges", type=int, default=12_000_000,
                   help="Max sum of n^2 pair-edges per forward pass (GRIT pads to full attention).")

    # internal
    p.add_argument("--stage", choices=["all", "analyze"], default="all",
                   help=argparse.SUPPRESS)

    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _strip_colab_kernel_args(argv)
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = str(args.drive_dir / "carriage_analysis")
    if args.dataset_dir is None:
        args.dataset_dir = str(args.drive_dir / "datasets")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Notebook-friendly entry point. Prefer main([...]) over bare main().

    Everything runs IN-PROCESS. A pasted Colab cell has no __file__, so there is no
    script path to relaunch as a subprocess; instead the Python-3.12/GRIT compatibility
    fixes are applied directly (apply_compat_patches, via prepare_inprocess) before GRIT
    is imported inside stage_analyze.
    """
    args = parse_args(argv)

    if args.stage == "analyze":
        # Direct call to the analysis stage (rare; the normal path falls through below
        # and calls stage_analyze after setup). Assumes the environment is already
        # prepared -- import grit must already work.
        stage_analyze(args)
        return

    mount_drive(args.drive_mount)
    args.drive_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        install_dependencies(args)
    else:
        log("[deps] Skipping dependency installation (--skip-install).")

    commit = clone_or_update_repo(args.repo_dir, args.repo_url, args.branch,
                                 args.commit or None, args.force_fresh_repo)
    run_cmd([sys.executable, "-m", "pip", "install", "-e", str(args.repo_dir)])
    validate_official_config(args.repo_dir, args.allow_upstream_config_drift)
    print_environment_summary(args.drive_dir, args.repo_dir, commit)

    ckpt, _epoch = find_checkpoint(args.drive_dir / "results", args.ckpt)
    args.ckpt = str(ckpt)

    # Prepare THIS process to import GRIT (compat patches + sys.path + cwd), then run the
    # analysis in-process. No subprocess, so this works from a pasted Colab cell.
    prepare_inprocess(args.repo_dir)
    log("\n[analyze] Running analysis in-process.")
    stage_analyze(args)

    log(f"\n[done] Figures + data: {args.out_dir}")


if __name__ == "__main__":
    main()
