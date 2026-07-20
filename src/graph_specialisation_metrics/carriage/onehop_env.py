"""1-hop GRIT env hooks: reconstruct the patched GRIT variants used for training.

The 1-hop variant is the official GRIT with a narrow patch that (a) writes the 1-hop ZINC
config, (b) adds a ``gt.attn.sparsity`` config key, and (c) makes GritTransformer select the
existing ``masked_rrwp_linear`` edge encoder when ``sparsity=one_hop`` -- restricting the
RRWP relative representation / attention support to original molecular bonds plus self,
instead of the complete graph. Model dimensions, parameter count (473,473), optimizer, and
schedule are unchanged.

We reuse the SAME patch function that produced the checkpoint
(experiments/zinc/training/grit_zinc_1hop_core.apply_parameter_matched_onehop_patch) so the
analysis model code matches training exactly -- essential for the checkpoint to load and the
1-hop masking to be active. It must run before GRIT is imported (it edits GRIT source files).

Carriage validity is unaffected: RRWP and the 1-hop mask are computed from the topology
(edge_index) only, so a semantic content swap still leaves the structure S -- including the
mask -- fixed (Def 3.2.1). The masking happens inside the edge encoder at forward time; the
per-graph Data objects (rrwp*, edge_index) are identical to the dense case.

The stricter ``zinc_1hop_local`` variant starts from that same 1-hop patch, then reuses
``grit_zinc_1hop_localrrwp_core.apply_parameter_matched_onehop_localrrwp_patch`` to zero
RRWP channels beyond identity + one step. Reusing the training patch here prevents the
analysis model from silently reconstructing a different architecture from its checkpoint.
"""

from __future__ import annotations

from pathlib import Path

from . import env
from .env import log


def apply_onehop_patch(repo_dir: Path, patch_note_dir: Path = None) -> None:
    env.ensure_repo_root_on_path()
    try:
        from experiments.zinc.training import grit_zinc_1hop_core as oh
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import experiments.zinc.training.grit_zinc_1hop_core. Ensure the "
            "PROJECT repo root is on sys.path (the bootstrap cell adds <repo>/src; the "
            "repo root holds the experiments/ package)."
        ) from exc
    note_dir = Path(patch_note_dir) if patch_note_dir is not None else Path(repo_dir)
    oh.apply_parameter_matched_onehop_patch(Path(repo_dir), note_dir)
    log("[1hop] applied parameter-matched 1-hop patch (masked RRWP edge encoder, "
        "sparsity=one_hop).")


def apply_onehop_localrrwp_patch(repo_dir: Path, patch_note_dir: Path = None) -> None:
    """Apply the exact 1-hop + local-only RRWP patch used to train the checkpoint."""
    env.ensure_repo_root_on_path()
    try:
        from experiments.zinc.training import grit_zinc_1hop_localrrwp_core as localrrwp
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import experiments.zinc.training.grit_zinc_1hop_localrrwp_core. "
            "Ensure the project repo root is on sys.path."
        ) from exc
    note_dir = Path(patch_note_dir) if patch_note_dir is not None else Path(repo_dir)
    localrrwp.apply_parameter_matched_onehop_localrrwp_patch(Path(repo_dir), note_dir)
    log("[1hop-local] applied parameter-matched 1-hop patch with local-only RRWP "
        "(identity + one-step channels; sparsity=one_hop_local_rrwp).")
