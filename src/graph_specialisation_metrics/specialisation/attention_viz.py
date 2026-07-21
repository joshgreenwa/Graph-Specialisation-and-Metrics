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


def _visual_bond_types(raw_types, dataset_format: str) -> np.ndarray:
    """Normalise task-specific categorical bonds to 1/2/3/aromatic=4 display codes."""
    values = np.asarray(raw_types, dtype=np.int64)
    return values + 1 if str(dataset_format) == "PyG-QM9" else values


def _spring_layout(n, bonds, seed=0):
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from([(int(a), int(b)) for a, b in bonds])
        # Molecules are small and connected; Kamada-Kawai gives a much more legible 2-D chemical
        # topology than a circular layout while remaining deterministic without RDKit/SMILES.
        pos = nx.kamada_kawai_layout(g)
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
        real_edges = (src < n) & (dest < n)
        has_virtual = bool(np.any(~real_edges))
        maps = {}
        for (l, h) in heads:
            a = cap["attn"][l][:, h].cpu().numpy()          # [E]
            A = np.zeros((n, n), dtype=np.float64)
            # VNode models append one virtual index. The molecule panel deliberately shows the
            # real-atom submatrix; its omitted mass is stated in the figure rather than indexing
            # the virtual row into an n-by-n atom matrix.
            A[dest[real_edges], src[real_edges]] = a[real_edges]  # A[i,j] = attention i<-j
            maps[(l, h)] = A
        raw_bonds = base.edge_index.cpu().numpy().T         # [2E, 2] directed bond pairs
        raw_types = getattr(base, "edge_attr", None)
        if raw_types is None:
            raw_types = np.ones(len(raw_bonds), dtype=np.int64)
        else:
            raw_types = raw_types.detach().cpu().numpy()
            raw_types = raw_types[:, 0] if raw_types.ndim > 1 else raw_types
        # PyG QM9 stores one-hot bond columns in the order single/double/triple/aromatic;
        # the training patch converts them to categorical indices 0..3. Normalise the cached
        # visualisation codes to 1..4 so they do not get confused with ZINC's bond-order-like
        # categories (1..3). Code 4 is rendered as aromatic/dashed by the comparison figures.
        raw_types = _visual_bond_types(
            raw_types, str(getattr(gm.cfg.dataset, "format", "")))
        bond_lookup = {}
        for edge, bond_type in zip(raw_bonds, raw_types):
            key = tuple(sorted((int(edge[0]), int(edge[1]))))
            bond_lookup.setdefault(key, int(bond_type))
        bonds = np.asarray(sorted(bond_lookup), dtype=np.int64)
        bond_types = np.asarray([bond_lookup[tuple(edge)] for edge in bonds], dtype=np.int64)
        # node colour = first content column (ZINC atom type; OGB atomic number) -- task-general.
        xc = base.x
        atom_types = (xc[:, 0] if xc.dim() > 1 else xc).cpu().numpy().astype(int)
        yv = base.y.reshape(-1)
        mols.append({
            "graph_id": int(gi), "n": n,
            "atom_types": atom_types,
            "bonds": bonds, "bond_types": bond_types,
            "pos": _spring_layout(n, bonds, seed=seed),
            "y": float(yv[0].item()), "y_dim": int(yv.numel()),
            "has_virtual": has_virtual, "maps": maps,
        })
    return {
        "molecules": mols,
        "heads": heads,
        "has_vnode": bool(any(m.get("has_virtual", False) for m in mols)),
        "atom_encoding": (
            "atomic_number"
            if getattr(gm.task, "node_content_desc", "") == "atomic number"
            else "category"
        ),
    }
