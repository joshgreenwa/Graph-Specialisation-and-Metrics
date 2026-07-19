"""Capture per-head attention maps across several molecules, for the interesting heads.

Pure visualisation (deliverable iii): for a chosen set of heads (l,h) and a few test molecules,
run one clean forward per molecule and read the softmax attention ``batch.attn`` into a dense
[n, n] matrix  A[i, j] = a^{lh}_{i<-j}  (row = destination/receiver i, col = source/sender j;
``edge_index[1]`` is the destination). We also carry the atom-type labels, the molecular bonds,
and a fixed spring layout so figures can render either the raw attention matrix or an
attention-weighted molecular graph. No intervention here -- this just shows *what the head reads*.
"""

from __future__ import annotations

import numpy as np


def _spring_layout(n, bonds, seed=0):
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from([(int(a), int(b)) for a, b in bonds])
        pos = nx.spring_layout(g, seed=seed, k=1.2 / max(np.sqrt(n), 1))
        return np.array([pos[i] for i in range(n)])
    except Exception:  # noqa: BLE001
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def collect_attention(gm, graph_ids, heads, seed=0) -> dict:
    """Return attention maps for each (molecule, head).

    Args:
        gm:        a loaded ``GritHeadModel``.
        graph_ids: iterable of eval-set graph indices (the molecules to show).
        heads:     iterable of (layer, head) tuples.

    Returns:
        dict(molecules=[...], heads=[...]) where molecules[g] has n, atom_types, bonds, pos, and
        maps[(l,h)] = [n, n] attention matrix.
    """
    from torch_geometric.data import Batch

    heads = [tuple(int(x) for x in hd) for hd in heads]
    mols = []
    for gi in graph_ids:
        base = gm.eval_ds[int(gi)]
        n = int(base.num_nodes)
        cb = Batch.from_data_list([base]).to(gm.device)
        cap = gm.capture(cb, want_grad=False, want_attn=True)
        ei = cap["edge_index"].cpu().numpy()               # [2, E] model support (dense=all pairs)
        src, dest = ei[0], ei[1]
        maps = {}
        for (l, h) in heads:
            a = cap["attn"][l][:, h].cpu().numpy()          # [E]
            A = np.zeros((n, n), dtype=np.float64)
            A[dest, src] = a                                # A[i, j] = attention i<-j
            maps[(l, h)] = A
        bonds = base.edge_index.cpu().numpy().T             # [2E, 2] directed bond pairs
        bonds = np.unique(np.sort(bonds, axis=1), axis=0)   # undirected unique
        # node colour = first content column (ZINC atom type; OGB atomic number) -- task-general.
        xc = base.x
        atom_types = (xc[:, 0] if xc.dim() > 1 else xc).cpu().numpy().astype(int)
        yv = base.y.reshape(-1)
        mols.append({
            "graph_id": int(gi), "n": n,
            "atom_types": atom_types,
            "bonds": bonds, "pos": _spring_layout(n, bonds, seed=seed),
            "y": float(yv[0].item()), "y_dim": int(yv.numel()), "maps": maps,
        })
    return {"molecules": mols, "heads": heads}
