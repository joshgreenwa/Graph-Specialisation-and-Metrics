"""Peptides-specific environment hooks, reusing the tested patches from the training code.

Peptides needs three things the ZINC path does not: RDKit (OGB SMILES featurization), a
dataset-loader compatibility patch (modern OGB import + non-interactive download), and a
streaming RRWP pre-transform (peptides graphs are large; the default in-memory collate can
OOM Colab RAM). All three use the same packaged helper functions exercised during training,
so analysis loads peptides exactly as training did.

Registered as GritTaskSpec.env_hooks so grit_runner applies them after cloning GRIT and
before building loaders.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from . import env
from .env import log, run_cmd


class _BaseShim:
    """The `base` interface the peptides patch helpers use: .log and .run_cmd."""

    log = staticmethod(log)

    @staticmethod
    def run_cmd(cmd, **kw):
        # The training helpers pass check=False sometimes; forward kwargs it understands.
        return run_cmd(cmd, check=kw.get("check", True))


def _peptides_common():
    """Import the exact training helpers from the installable internal runtime."""
    try:
        from ..grit_patches import peptides as pc
        return pc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import the packaged checkpoint-compatible Peptides GRIT patches."
        ) from exc


def install_peptides_deps() -> None:
    """RDKit for OGB SMILES featurization (idempotent)."""
    _peptides_common().install_peptides_dependencies(_BaseShim)


def apply_peptides_patches(repo_dir: Path) -> None:
    """Dataset-loader compat + streaming RRWP patches on the cloned GRIT repo."""
    pc = _peptides_common()
    pc.apply_peptides_dataset_compat_patch(_BaseShim, repo_dir)
    pc.apply_peptides_streaming_rrwp_patch(_BaseShim, repo_dir)
    # The streaming patch reads this to bound peak RAM during RRWP collation.
    os.environ.setdefault("GRIT_PE_STREAM_CHUNK_SIZE", "32")
    log(f"[peptides] GRIT_PE_STREAM_CHUNK_SIZE={os.environ['GRIT_PE_STREAM_CHUNK_SIZE']}")


def apply_peptides_struct_onehop_patch(repo_dir: Path) -> None:
    """Reconstruct the exact sparse Peptides-struct model used for training.

    The source intervention is the shared masked-RRWP 1-hop patch.  Its training
    helper is parameterised through module constants, so set those constants to
    the Peptides config only for the duration of the patch and restore them
    afterwards.  This keeps repeated dense/1-hop preparation in one process
    deterministic.
    """

    from ..grit_patches import peptides as pc
    from ..grit_patches import zinc_onehop as onehop

    previous = (
        onehop.DENSE_OFFICIAL_CFG,
        onehop.OFFICIAL_CFG,
        onehop.ONE_HOP_CFG_TEXT,
    )
    try:
        onehop.DENSE_OFFICIAL_CFG = pc.DENSE_OFFICIAL_CFG
        onehop.OFFICIAL_CFG = pc.ONEHOP_OFFICIAL_CFG
        onehop.ONE_HOP_CFG_TEXT = pc.PEPTIDES_STRUCT_ONEHOP_CFG_TEXT
        onehop.apply_parameter_matched_onehop_patch(Path(repo_dir), Path(repo_dir))
    finally:
        (
            onehop.DENSE_OFFICIAL_CFG,
            onehop.OFFICIAL_CFG,
            onehop.ONE_HOP_CFG_TEXT,
        ) = previous
    log("[peptides] applied checkpoint-compatible Peptides-struct 1-hop patch")


def apply_peptides_variant_patch(
    repo_dir: Path,
    *,
    dataset: str,
    attention: str,
    hops: int,
    global_vnode: bool,
) -> None:
    """Replay the unified Peptides training patch and materialise its exact config."""

    if dataset not in {"func", "struct"}:
        raise ValueError(f"unsupported Peptides dataset {dataset!r}")
    if attention not in {"dense", "khop"}:
        raise ValueError(f"unsupported Peptides attention mode {attention!r}")
    if int(hops) not in {1, 2}:
        raise ValueError(f"unsupported Peptides hop count {hops!r}")

    ensure_repo_root_on_path()
    from experiments.peptides.training import GRIT_peptides_khop as training

    training.apply_peptides_pandas_warning_patch(repo_dir)
    if dataset == "func":
        training.apply_peptides_multilabel_metric_patch(repo_dir)
    training.apply_attention_rrwp_vnode_patch(repo_dir)
    training.verify_attention_rrwp_vnode_patch(repo_dir)
    args = SimpleNamespace(
        task=dataset,
        attention=attention,
        hops=int(hops),
        global_vnode=bool(global_vnode),
        rrwp_horizon=-1,
        wandb_project="",
    )
    training.make_run_config(repo_dir, args)
    training.validate_config(repo_dir, args, allow_drift=False)
    log(
        "[peptides] applied unified checkpoint-compatible patch: "
        f"dataset={dataset}, attention={attention}, hops={int(hops)}, "
        f"global_vnode={bool(global_vnode)}"
    )


def ensure_repo_root_on_path(repo_dir: Path = None) -> None:
    """Retained compatibility hook for callers that also need repository-local resources."""
    env.ensure_repo_root_on_path()
