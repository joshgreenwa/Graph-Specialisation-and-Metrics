"""Content adapters: the task-specific half of a *semantic* intervention.

A semantic intervention (Def 3.2.1) replaces one node's content while holding structure S
fixed. What "content" is, and how it is written into a batched tensor, depends on the
model's node encoder. This module abstracts that so the analysis loop in ``grit_runner``
stays task-agnostic.

For GRIT the swappable content is exactly ``data.x`` -- the integer node features the node
encoder consumes -- and RRWP lives separately in ``data.rrwp*`` (a pre-transform of the
topology). So a single ``FullNodeContentAdapter`` that swaps whole ``x`` rows covers every
GRIT node encoder:

  * ZINC / TypeDictNode:  x is [n, 1], one atom-type integer per node.
  * Peptides / OGB Atom:  x is [n, 9], the nine OGB atom features per node.

A donor swap replaces node j's entire feature row with a real donor node's row sampled
from another graph, which keeps the perturbed graph on-manifold (Def 3.2.2).
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class ContentAdapter(Protocol):
    """How to read and overwrite the swappable node content (whole feature rows)."""

    def rows(self, data) -> np.ndarray:
        """[n, F] int array: the swappable content rows of one graph (== data.x)."""
        ...

    def num_symbols(self, cfg) -> Optional[int]:
        """Vocabulary size for an optional range check (None = skip; e.g. multi-field Atom)."""
        ...

    def write_donors(self, batch_x, row_idx, donor_rows) -> None:
        """In place: batch_x[row_idx] = donor_rows (the swapped node content per replica)."""
        ...


class FullNodeContentAdapter:
    """Swap a node's whole ``x`` feature row. Works for every GRIT node encoder.

    Reading/writing the full row is exactly the semantic intervention: for TypeDictNode the
    row is a single atom-type integer, for OGB's Atom encoder it is the nine atom features.
    Both are consumed verbatim by the encoder, and RRWP (structure) is untouched.
    """

    def __init__(self, num_symbols: Optional[int] = None):
        self._num_symbols = num_symbols

    def rows(self, data) -> np.ndarray:
        x = data.x
        if x.dim() == 1:
            x = x.view(-1, 1)
        return x.cpu().numpy().astype(np.int64)

    def num_symbols(self, cfg) -> Optional[int]:
        if self._num_symbols is not None:
            return self._num_symbols
        # TypeDictNode exposes a single vocabulary size; multi-field Atom does not.
        n = getattr(getattr(cfg, "dataset", object()), "node_encoder_num_types", 0)
        return int(n) if n and int(n) > 0 else None

    def write_donors(self, batch_x, row_idx, donor_rows) -> None:
        import torch

        vals = torch.as_tensor(donor_rows, device=batch_x.device, dtype=batch_x.dtype)
        if batch_x.dim() == 1:
            batch_x[row_idx] = vals.view(-1)
        else:
            batch_x[row_idx] = vals.view(len(row_idx), -1)


# Backwards-compatible name: ZINC used "TypeDict"; whole-row swap is identical for [n,1] x.
TypeDictContentAdapter = FullNodeContentAdapter
