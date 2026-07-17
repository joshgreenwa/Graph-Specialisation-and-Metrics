"""Content adapters: the task-specific half of a *semantic* intervention.

A semantic intervention (Def 3.2.1) replaces one node's content while holding structure
S fixed. What "content" is, and how it is written into a batched tensor, depends on the
model's node encoder. This module abstracts that so the analysis loop in ``grit_runner``
stays task-agnostic.

The default ``TypeDictContentAdapter`` covers every GRIT task whose node encoder is
``TypeDictNode`` -- a single integer symbol per node stored in ``data.x[:, 0]`` (ZINC atom
type, and most molecular GRIT configs). A task with a different encoder (e.g. OGB's
multi-field AtomEncoder) supplies its own adapter with the same three methods.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ContentAdapter(Protocol):
    """How to read, count, and overwrite the swappable node content."""

    def symbols(self, data) -> np.ndarray:
        """[n] int array: the swappable symbolic content per node of one graph."""
        ...

    def num_symbols(self, cfg) -> int:
        """Vocabulary size, for donor-range validation."""
        ...

    def write_donors(self, batch_x, rows, donor_vals) -> None:
        """In place: set the swapped node content for each replica.

        ``batch_x`` is the collated node-feature tensor, ``rows`` the flat row indices of
        the perturbed nodes (one per replica), ``donor_vals`` the donor symbols to write.
        """
        ...

    def unchanged_columns(self, batch_x):
        """Columns of batch_x that a swap must leave untouched (for the structure check).

        Returns None to mean "the whole tensor except the written cells"; the structure
        check then compares whole rows.
        """
        ...


class TypeDictContentAdapter:
    """Single integer symbol per node at ``x[:, 0]`` (GRIT's TypeDictNode encoder).

    This is exactly what the encoder consumes: ``TypeDictNodeEncoder.forward`` does
    ``self.encoder(batch.x[:, 0])``. Swapping ``x[:, 0]`` is therefore the minimal,
    on-manifold content edit, and every other column (if any) is inert.
    """

    def symbols(self, data) -> np.ndarray:
        return data.x[:, 0].cpu().numpy().astype(np.int64)

    def num_symbols(self, cfg) -> int:
        return int(cfg.dataset.node_encoder_num_types)

    def write_donors(self, batch_x, rows, donor_vals) -> None:
        import torch

        batch_x[rows, 0] = torch.as_tensor(donor_vals, device=batch_x.device, dtype=batch_x.dtype)

    def unchanged_columns(self, batch_x):
        # Only column 0 carries the symbol; the encoder ignores the rest. We still verify
        # the whole row is unchanged except at the written cells (None => full-row check),
        # which is the strictest and matches the original ZINC runner.
        return None
