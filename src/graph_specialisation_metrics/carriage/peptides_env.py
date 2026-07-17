"""Peptides-specific environment hooks, reusing the tested patches from the training code.

Peptides needs three things the ZINC path does not: RDKit (OGB SMILES featurization), a
dataset-loader compatibility patch (modern OGB import + non-interactive download), and a
streaming RRWP pre-transform (peptides graphs are large; the default in-memory collate can
OOM Colab RAM). All three already exist and are exercised by the training runner in
``experiments/peptides_struct/training/grit_peptides_struct_common.py``; we call them here
with a tiny shim so the analysis loads peptides EXACTLY as training did -- no duplicated,
drifting patch logic.

Registered as GritTaskSpec.env_hooks so grit_runner applies them after cloning GRIT and
before building loaders.
"""

from __future__ import annotations

import os
from pathlib import Path

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
    """Import the training repo's peptides helper module (must be on sys.path already)."""
    try:
        from experiments.peptides_struct.training import grit_peptides_struct_common as pc
        return pc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import experiments.peptides_struct.training.grit_peptides_struct_common. "
            "Ensure <repo>/src AND <repo> are on sys.path (the bootstrap cell adds <repo>/src; "
            "the repo root holds the experiments/ package)."
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


def ensure_repo_root_on_path(repo_dir: Path = None) -> None:
    """Put the PROJECT repo root on sys.path so `import experiments...` resolves."""
    env.ensure_repo_root_on_path()
