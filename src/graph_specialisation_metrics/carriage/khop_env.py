"""k-hop / global-VNode GRIT env hooks: reconstruct the patched GRIT used for training.

The 2-hop and virtual-node ZINC controls were trained with ``apply_khop_patch``, which
(a) writes the k-hop ZINC config
(``configs/GRIT/zinc-GRIT-RRWP-khop.yaml`` with ``gt.attn.sparsity=k_hop``,
``gt.attn.hops=k``, ``gt.attn.global_vnode=<bool>``), and (b) patches the official GRIT
source so that a masked RRWP edge encoder restricts the relative representation / attention
support to ordered node pairs at shortest-path distance <= k (self included), and, when
requested, adds one learned ``GlobalVNode`` per graph that attends bidirectionally with every
real node in every layer and is excluded from final pooling.

We reuse the same packaged ``apply_khop_patch`` function that produced the checkpoints, so the
analysis model code matches training exactly --
essential for the checkpoint to load and the k-hop masking / VNode to be active. It must run
before GRIT is imported (it edits GRIT source files).

Carriage validity is unaffected. RRWP, the k-hop mask, and the VNode's degree feature are all
computed from the topology (edge_index) only, so a semantic content swap leaves the structure S
-- including the mask and the VNode wiring -- fixed (Def 3.2.1). The VNode is a communication-
only token removed before pooling (``batch.x = batch.x[batch.real_node_mask]``), so the readout
is still add-pooling over the real nodes and the shared-gradient precondition holds; this is
asserted at run time by ``grit_runner.check_carriage_preconditions``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from .env import log


def _import_khop_patch():
    """Import the exact training patch from the installable internal runtime."""
    try:
        from ..grit_patches.khop_zinc import apply_khop_patch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import the packaged checkpoint-compatible k-hop GRIT patch."
        ) from exc
    return apply_khop_patch


def make_khop_hook(hops: int, global_vnode: bool) -> Callable[[Path], None]:
    """Return a ``hook(repo_dir)`` that replays the exact k-hop (+ optional VNode) patch."""

    def hook(repo_dir: Path) -> None:
        apply_khop_patch = _import_khop_patch()
        # Reconstruct the relevant subset of the training runner's parsed arguments.  These
        # analysis tasks are always k-hop controls; ``wandb_project=None`` preserves the
        # training patch's default ZINC project label.  ``expected_params=None`` keeps the
        # automatic guard (473473 without VNode, +64 with VNode).  The drive_dir argument is
        # used only to drop a provenance note; point it at the clone so we never write to the
        # user's Drive.
        args = argparse.Namespace(
            attention="khop",
            hops=int(hops),
            global_vnode=bool(global_vnode),
            expected_params=None,
            wandb_project=None,
        )
        apply_khop_patch(Path(repo_dir), Path(repo_dir), args)
        log(f"[khop] applied k-hop patch (hops={hops}, global_vnode={global_vnode}; "
            "masked RRWP edge encoder, sparsity=k_hop).")

    return hook
