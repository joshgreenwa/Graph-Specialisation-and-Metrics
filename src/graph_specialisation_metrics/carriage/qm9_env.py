"""QM9 gap GRIT environment hooks: replay the checkpoint-compatible training patch.

The dense and 1-hop checkpoints were produced by the repository-root
``GRIT_QM9_gap.py`` runner.  That runner patches the pinned official GRIT checkout with
the QM9 loader (target column 4 in eV, atomic-number node content, categorical bonds and
the seeded 110000/10000/remainder split) plus dense/configurable-k-hop attention support.

Analysis must replay that patch before importing GRIT. Reusing the training function avoids a
second loader/model reconstruction. Its one-hop encoder has one parameter-neutral analysis
refinement: clean support is still exactly bonds+self, but it is read from ``edge_index`` so the
mask-frozen RRWP intervention cannot silently rewire attention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from . import env
from .env import log


def _import_qm9_patch():
    """Import the patch and verifier from the repository-root training runner."""
    env.ensure_repo_root_on_path()
    try:
        from GRIT_QM9_gap import apply_qm9_patch, verify_qm9_model_patch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import the QM9 patch from the repository-root GRIT_QM9_gap.py. "
            "The Colab bootstrap must put both <repo>/src and <repo> on sys.path."
        ) from exc
    return apply_qm9_patch, verify_qm9_model_patch


def make_qm9_hook(
    attention: str,
    hops: int = 1,
    *,
    global_vnode: bool = False,
) -> Callable[[Path], None]:
    """Return a hook reconstructing either dense or exact ``<=k``-hop QM9 GRIT."""
    if attention not in {"dense", "khop"}:
        raise ValueError(f"attention must be 'dense' or 'khop', got {attention!r}")
    if not 1 <= int(hops) <= 20:
        raise ValueError("hops must be in [1, 20] for the RRWP-21 QM9 configuration")

    def hook(repo_dir: Path) -> None:
        apply_qm9_patch, verify_qm9_model_patch = _import_qm9_patch()
        args = argparse.Namespace(
            attention=str(attention),
            hops=int(hops),
            global_vnode=bool(global_vnode),
            batch_size=128,
            epochs=300,
            warmup_epochs=10,
            # The training runner's provenance note calls expected_param_count(args).
            expected_params=None,
        )
        # ``drive_dir`` is used only for a patch provenance note. Keep analysis writes inside
        # the disposable task-specific clone rather than touching a training directory.
        apply_qm9_patch(Path(repo_dir), Path(repo_dir), args)
        verify_qm9_model_patch(Path(repo_dir))
        variant = "dense" if attention == "dense" else f"{int(hops)}-hop"
        if global_vnode:
            variant += "+VNode"
        log(
            f"[qm9] applied checkpoint-compatible QM9-gap {variant} patch "
            "(target=column 4 eV; split=110000/10000/remainder, seed=42)."
        )

    return hook
