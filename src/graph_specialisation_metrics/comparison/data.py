"""Cache paths, loaders, and method metadata for the cross-model comparison deliverables.

Pure numpy/stdlib (no torch, no GRIT) so the plotting deliverables can be rebuilt from the
cached artefacts without a GPU or a re-run. The cache is exactly what the existing carriage and
specialisation orchestrators already write to Drive:

* carriage (per task, per intervention): ``<carriage_collate>/<task>/[<prefix>]carriage_summary.json``
  where ``<prefix>`` is ``""`` for semantic and ``"structural_<mode>_"`` for structural. Each
  holds a ``curves`` block (F/B/S/B_far arrays keyed by SPD ``bin_label``) plus a ``meta`` block
  (checkpoint, val/test metric, num_graphs, donors_K, ...).
* specialisation (per task): ``<spec_collate>/<task>/scores_<task>.npz`` with the per-head
  ``S_sem`` / ``S_str`` / ``S_attn_sem`` ``[L, H]`` matrices, and ``stats_<task>.json`` with the
  val/test metrics + checks.

``select_methods`` / ``is_vnode`` implement the drop/include filtering the deliverables need
(e.g. drop the VNode runs for a final figure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# The five ZINC GRIT models this comparison targets, in a canonical display order.
DEFAULT_TASKS = ["zinc", "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"]

# Drive collate roots (mirror carriage.colab.DEFAULT_COLLATE_DIR and
# specialisation.colab.DEFAULT_COLLATE_DIR). Kept here in the torch-free layer so the figure
# builder can locate the cache and be imported/run without a GPU.
CARRIAGE_COLLATE_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/carriage_figures"
SPEC_COLLATE_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/specialisation_figures"
COMPARISON_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/model_comparison"

# Per-method display metadata: short label, colour, marker, and family (for drop/include).
# Tasks absent here fall back to a neutral style + the task name as its label.
METHOD_META = {
    "zinc":            dict(label="Dense",         color="#444444", marker="o", family="dense"),
    "zinc_1hop":       dict(label="1-hop",         color="#1f77b4", marker="^", family="sparse"),
    "zinc_2hop":       dict(label="2-hop",         color="#2ca02c", marker="s", family="sparse"),
    "zinc_1hop_local": dict(label="1-hop (local RRWP)", color="#17becf", marker="<", family="sparse"),
    "zinc_1hop_vnode": dict(label="1-hop + VNode", color="#ff7f0e", marker="v", family="vnode"),
    "zinc_2hop_vnode": dict(label="2-hop + VNode", color="#d62728", marker="D", family="vnode"),
}
_FALLBACK_PALETTE = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def method_meta(task: str, i: int = 0) -> dict:
    """Display metadata for a task, falling back to a neutral style for unknown tasks."""
    if task in METHOD_META:
        return METHOD_META[task]
    return dict(label=task, color=_FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)],
                marker="o", family="other")


def is_vnode(task: str) -> bool:
    """True for the virtual-node variants (used by ``drop_vnode``)."""
    return "vnode" in task.lower()


def select_methods(tasks: Sequence[str], *, include: Optional[Sequence[str]] = None,
                   exclude: Optional[Sequence[str]] = None, drop_vnode: bool = False) -> list:
    """Filter a task list for a figure.

    ``include`` restricts to a whitelist (order follows ``tasks``); ``exclude`` drops a blacklist;
    ``drop_vnode`` additionally drops every VNode variant. Order is preserved from ``tasks``.
    """
    out = list(tasks)
    if include is not None:
        inc = set(include)
        out = [t for t in out if t in inc]
    if exclude:
        exc = set(exclude)
        out = [t for t in out if t not in exc]
    if drop_vnode:
        out = [t for t in out if not is_vnode(t)]
    return out


# --------------------------------------------------------------------------------------
# Cache paths
# --------------------------------------------------------------------------------------

def carriage_prefix(intervention: str, structural_mode: str = "transposition") -> str:
    """Filename prefix the carriage figures module uses (empty for semantic)."""
    if intervention == "semantic":
        return ""
    if intervention == "structural":
        return f"structural_{structural_mode}_"
    raise ValueError(f"intervention must be 'semantic' or 'structural', got {intervention!r}")


def carriage_summary_path(carriage_collate, task: str, intervention: str,
                          structural_mode: str = "transposition") -> Path:
    prefix = carriage_prefix(intervention, structural_mode)
    return Path(carriage_collate) / task / f"{prefix}carriage_summary.json"


def carriage_pairs_path(carriage_collate, task: str, intervention: str,
                        structural_mode: str = "transposition") -> Path:
    prefix = carriage_prefix(intervention, structural_mode)
    return Path(carriage_collate) / task / f"{prefix}carriage_pairs.npz"


def scores_npz_path(spec_collate, task: str) -> Path:
    return Path(spec_collate) / task / f"scores_{task}.npz"


def channel_ablation_npz_path(spec_collate, task: str) -> Path:
    return Path(spec_collate) / task / f"channel_ablation_{task}.npz"


def spec_stats_path(spec_collate, task: str) -> Path:
    return Path(spec_collate) / task / f"stats_{task}.json"


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

def load_carriage_summary(path) -> Optional[dict]:
    """Read a carriage_summary.json (curves + meta), or None if it is missing."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_scores(path) -> Optional[dict]:
    """Read a scores_<task>.npz -> {S_sem, S_str, S_attn_sem} as float arrays, or None."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as z:
        out = {"S_sem": np.asarray(z["S_sem"], float), "S_str": np.asarray(z["S_str"], float)}
        if "S_attn_sem" in z:
            out["S_attn_sem"] = np.asarray(z["S_attn_sem"], float)
    return out


def load_spec_stats(path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_carriage_curves_by_task(carriage_collate, tasks: Sequence[str], intervention: str,
                                 structural_mode: str = "transposition") -> dict:
    """{task: summary_dict} for every task whose carriage summary exists on disk."""
    out = {}
    for t in tasks:
        s = load_carriage_summary(
            carriage_summary_path(carriage_collate, t, intervention, structural_mode))
        if s is not None:
            out[t] = s
    return out


def load_scores_by_task(spec_collate, tasks: Sequence[str]) -> dict:
    """{task: scores_dict} for every task whose scores npz exists on disk."""
    out = {}
    for t in tasks:
        s = load_scores(scores_npz_path(spec_collate, t))
        if s is not None:
            out[t] = s
    return out


# channel-split causal ablation (I_sem/I_str functional + loss, and single-channel overall_*).
_CHANNEL_ABLATION_KEYS = ("I_sem_func", "I_str_func", "I_sem_loss", "I_str_loss",
                          "overall_func", "overall_loss")


def load_channel_ablation(path) -> Optional[dict]:
    """Read a channel_ablation_<task>.npz -> {I_sem_func, I_str_func, ...} float [L,H] arrays, or None."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: np.asarray(z[k], float) for k in _CHANNEL_ABLATION_KEYS if k in z}


def load_channel_ablation_by_task(spec_collate, tasks: Sequence[str]) -> dict:
    """{task: channel_ablation_dict} for every task whose channel_ablation npz exists on disk."""
    out = {}
    for t in tasks:
        c = load_channel_ablation(channel_ablation_npz_path(spec_collate, t))
        if c is not None:
            out[t] = c
    return out


# --------------------------------------------------------------------------------------
# Curve extraction helpers (align methods on a shared SPD-bin axis)
# --------------------------------------------------------------------------------------

def master_bins(curves_by_task: dict, tasks: Sequence[str]) -> list:
    """Ordered union of SPD bin labels across the given tasks (ascending by lower edge).

    ZINC's small diameter means the log-strategy bins usually coincide across models, but this
    tolerates differing ``dmax`` by ordering the union on each label's lower edge.
    """
    lo_of: dict = {}
    for t in tasks:
        c = curves_by_task.get(t)
        if not c:
            continue
        cur = c["curves"]
        for lab, lo in zip(cur["bin_label"], cur["bin_lo"]):
            lo_of.setdefault(str(lab), int(lo))
    return [lab for lab, _ in sorted(lo_of.items(), key=lambda kv: kv[1])]


def curve_on_master(curve: dict, key: str, master: Sequence[str]) -> np.ndarray:
    """A curve array (e.g. 'F_mean') re-indexed onto the master bin-label axis (NaN where absent)."""
    label_to_val = {str(lab): v for lab, v in zip(curve["bin_label"], curve[key])}
    return np.array([label_to_val.get(lab, np.nan) for lab in master], dtype=float)
