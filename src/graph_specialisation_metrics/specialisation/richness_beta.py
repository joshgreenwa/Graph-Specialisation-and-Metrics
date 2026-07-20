"""BETA: response-dimensionality extension of the ZINC per-head specialisation scores.

This is intentionally an isolated experiment. It uses the production scorer's existing
semantic donor swaps, mask-frozen structural transpositions, clean readout gradients, and
per-head transport deltas. No channel is ablated and no additional intervention is introduced.

The ordinary score collapses every source/carrier response into one magnitude per head. Here we
retain the immediately preceding tensor

    A[channel, source, distance-bin, layer, head]
      = sum_{carrier in bin} F^{layer,head}[carrier, source]

and ask whether different sources recruit proportionally different head patterns. Each source
row is L2-normalised (removing total strength), and the participation ratio of the resulting
head-pattern second moment measures readout-relevant response dimensionality. It is invariant to
head permutation and overall response scale. It is not claimed to be information-theoretic rank.

The predeclared primary contrast is standard 1-hop/global-RRWP minus 1-hop/local-RRWP structural
response rank, graph-paired by using identical ZINC graphs, sources, donors, and partners. A
semantic contrast is the matched specificity control. If the paired structural interval includes
zero, the experiment reports a null/inconclusive result rather than promoting the metric.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..carriage import env
from ..carriage.env import log
from ..carriage.grit_runner import _spd
from ..carriage.tasks import GritTaskSpec, get_task
from .model import SpecConfig
from .scores import score_model


BETA_TASKS = ("zinc", "zinc_1hop", "zinc_1hop_local")
DEFAULT_OUT_DIR = (
    "/content/drive/MyDrive/graph_specialisation_metrics/beta_zinc_head_richness"
)
BIN_LABELS = ("0", "1", "2", "3", "4-7", "8+")


def _distance_bin_masks(distances: np.ndarray) -> list[np.ndarray]:
    """Fixed, cross-model ZINC bins over a [sources, carriers] SPD matrix."""
    d = np.asarray(distances, dtype=float)
    return [
        d == 0,
        d == 1,
        d == 2,
        d == 3,
        (d >= 4) & (d <= 7),
        (d >= 8) & np.isfinite(d),
    ]


@dataclass
class ResponseCollector:
    """Observe scorer intermediates without changing its intervention or estimator."""

    task: str
    records: dict[str, list[dict]] = field(
        default_factory=lambda: {"semantic": [], "structural": []}
    )
    abs_error: dict[int, float] = field(default_factory=dict)
    _spd_cache: dict[int, np.ndarray] = field(default_factory=dict)

    def __call__(
        self,
        *,
        channel,
        graph_id,
        base,
        source_nodes,
        phi_stack,
        donor_averaged_delta,
        clean_prediction,
    ) -> None:
        import torch

        if channel not in self.records:
            raise ValueError(f"unknown response channel {channel!r}")
        source_nodes = np.asarray(source_nodes, dtype=np.int64)
        n = int(base.num_nodes)
        if graph_id not in self._spd_cache:
            self._spd_cache[graph_id] = _spd(base, n)
        # D is carrier x source; transpose selected columns to source x carrier.
        distances = self._spd_cache[graph_id][:, source_nodes].T
        masks = [
            torch.as_tensor(m, device=phi_stack[0].device, dtype=phi_stack[0].dtype)
            for m in _distance_bin_masks(distances)
        ]

        S, L, H = len(source_nodes), len(phi_stack), int(phi_stack[0].shape[2])
        response = torch.zeros(
            S, len(BIN_LABELS), L, H,
            device=phi_stack[0].device, dtype=phi_stack[0].dtype,
        )
        for layer, (phi, delta) in enumerate(zip(phi_stack, donor_averaged_delta)):
            # Same functional-magnitude quantity as the production specialisation score.
            projected = torch.einsum("tnhd,snhd->tsnh", phi, delta)
            functional = (projected * projected).sum(dim=0).sqrt()  # [S,n,H]
            for b, mask in enumerate(masks):
                response[:, b, layer, :] = torch.einsum(
                    "snh,sn->sh", functional, mask
                )

        self.records[channel].append({
            "graph_id": int(graph_id),
            "source_nodes": source_nodes.copy(),
            "response": response.detach().cpu().numpy().reshape(
                S, len(BIN_LABELS), L * H
            ).astype(np.float32),
        })
        if channel == "semantic":
            pred = float(clean_prediction.detach().cpu().reshape(-1)[0])
            target = float(base.y.detach().cpu().reshape(-1)[0])
            self.abs_error[int(graph_id)] = abs(pred - target)

    def score_reconstruction(self, channel: str) -> np.ndarray:
        """Rebuild S_channel from collected distance bins (audit against production score)."""
        records = self.records[channel]
        if not records:
            raise ValueError(f"no {channel} responses collected")
        total = sum(r["response"].sum(axis=(0, 1)) for r in records)
        count = sum(int(r["response"].shape[0]) for r in records)
        return total / max(count, 1)


def participation_ratio(second_moment: np.ndarray) -> float:
    """Effective rank (inverse spectral concentration) of a PSD second-moment matrix."""
    M = np.asarray(second_moment, dtype=np.float64)
    trace = float(np.trace(M))
    denom = float(np.square(M).sum())
    if trace <= 0.0 or denom <= 0.0:
        return float("nan")
    return trace * trace / denom


def _activity_scale(records: list[dict]) -> float:
    masses = np.concatenate([r["response"].sum(axis=(1, 2)) for r in records])
    positive = masses[masses > 0]
    return float(np.median(positive)) if positive.size else 0.0


def _graph_moments(
    records: list[dict],
    bin_index: Optional[int],
    activity_floor: float,
) -> dict[int, dict]:
    """Per-graph second moments after source-wise amplitude normalisation."""
    out = {}
    for record in records:
        response = np.asarray(record["response"], dtype=np.float64)
        X = response.sum(axis=1) if bin_index is None else response[:, bin_index, :]
        mass = X.sum(axis=1)
        active = mass > float(activity_floor)
        X = X[active]
        if X.size:
            norms = np.linalg.norm(X, axis=1)
            keep = norms > 0
            X = X[keep] / norms[keep, None]
        p = int(response.shape[-1])
        gram = X.T @ X if X.size else np.zeros((p, p), dtype=np.float64)
        out[int(record["graph_id"])] = {
            "gram": gram,
            "active_sources": int(X.shape[0]) if X.ndim == 2 else 0,
            "total_sources": int(response.shape[0]),
        }
    return out


def _sum_grams(per_graph: dict[int, dict], graph_ids) -> np.ndarray:
    ids = list(graph_ids)
    if not ids:
        p = next(iter(per_graph.values()))["gram"].shape[0]
        return np.zeros((p, p), dtype=np.float64)
    return np.add.reduce([per_graph[int(g)]["gram"] for g in ids])


def _rank_summary(per_graph: dict[int, dict], graph_ids) -> dict:
    ids = [int(g) for g in graph_ids]
    gram = _sum_grams(per_graph, ids)
    active = sum(per_graph[g]["active_sources"] for g in ids)
    total = sum(per_graph[g]["total_sources"] for g in ids)
    return {
        "rank": participation_ratio(gram),
        "active_sources": int(active),
        "total_sources": int(total),
        "active_fraction": float(active / max(total, 1)),
        "active_graphs": int(sum(per_graph[g]["active_sources"] > 0 for g in ids)),
    }


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def analyse_response_rank(
    task_results: dict,
    *,
    n_boot: int = 1000,
    bootstrap_seed: int = 1729,
    activity_floor_fraction: float = 1e-4,
) -> dict:
    """Graph-paired rank estimates, global/local contrasts, and performance linkage."""
    tasks = list(task_results)
    graph_sets = [set(map(int, task_results[t]["graph_ids"])) for t in tasks]
    common = sorted(set.intersection(*graph_sets))
    if not common:
        raise RuntimeError("the three tasks have no common scored graph ids")
    if any(set(common) != s for s in graph_sets):
        raise RuntimeError("paired beta analysis requires identical graph selections")

    rng = np.random.default_rng(bootstrap_seed)
    boot_ids = [
        rng.choice(common, size=len(common), replace=True).astype(np.int64)
        for _ in range(int(n_boot))
    ]
    keys = list(BIN_LABELS) + ["overall"]
    moments: dict = {}
    summaries: dict = {}
    bootstrap: dict = {}

    for task in tasks:
        moments[task] = {}
        summaries[task] = {}
        bootstrap[task] = {}
        collector: ResponseCollector = task_results[task]["collector"]
        for channel in ("semantic", "structural"):
            records = collector.records[channel]
            scale = _activity_scale(records)
            floor = float(activity_floor_fraction) * scale
            moments[task][channel] = {}
            summaries[task][channel] = {}
            bootstrap[task][channel] = {}
            for z, key in enumerate(keys):
                bin_index = None if key == "overall" else z
                pg = _graph_moments(records, bin_index, floor)
                moments[task][channel][key] = pg
                summary = _rank_summary(pg, common)
                draws = np.asarray([
                    participation_ratio(_sum_grams(pg, ids)) for ids in boot_ids
                ], dtype=float)
                finite = draws[np.isfinite(draws)]
                summary.update({
                    "ci_low": float(np.quantile(finite, 0.025)) if finite.size else None,
                    "ci_high": float(np.quantile(finite, 0.975)) if finite.size else None,
                    "activity_floor": floor,
                })
                summaries[task][channel][key] = summary
                bootstrap[task][channel][key] = draws

    global_task, local_task = "zinc_1hop", "zinc_1hop_local"
    contrasts = {}
    for key in keys:
        contrasts[key] = {}
        for channel in ("semantic", "structural"):
            draws = (
                bootstrap[global_task][channel][key]
                - bootstrap[local_task][channel][key]
            )
            finite = draws[np.isfinite(draws)]
            point = (
                summaries[global_task][channel][key]["rank"]
                - summaries[local_task][channel][key]["rank"]
            )
            contrasts[key][channel] = {
                "difference": float(point),
                "ci_low": float(np.quantile(finite, 0.025)) if finite.size else None,
                "ci_high": float(np.quantile(finite, 0.975)) if finite.size else None,
                "bootstrap": draws,
            }
        double_draws = (
            contrasts[key]["structural"]["bootstrap"]
            - contrasts[key]["semantic"]["bootstrap"]
        )
        finite = double_draws[np.isfinite(double_draws)]
        contrasts[key]["structural_minus_semantic"] = {
            "difference": float(
                contrasts[key]["structural"]["difference"]
                - contrasts[key]["semantic"]["difference"]
            ),
            "ci_low": float(np.quantile(finite, 0.025)) if finite.size else None,
            "ci_high": float(np.quantile(finite, 0.975)) if finite.size else None,
            "bootstrap": double_draws,
        }

    # Per-graph confirmatory association: does a graph's global-RRWP structural-rank gain
    # accompany its absolute-error improvement over the local-RRWP checkpoint?
    pg_global = moments[global_task]["structural"]["overall"]
    pg_local = moments[local_task]["structural"]["overall"]
    rank_gain, error_gain, used_graphs = [], [], []
    global_errors = task_results[global_task]["collector"].abs_error
    local_errors = task_results[local_task]["collector"].abs_error
    for gid in common:
        rg = participation_ratio(pg_global[gid]["gram"])
        rl = participation_ratio(pg_local[gid]["gram"])
        if np.isfinite(rg) and np.isfinite(rl):
            used_graphs.append(gid)
            rank_gain.append(rg - rl)
            error_gain.append(local_errors[gid] - global_errors[gid])
    rho = _spearman(rank_gain, error_gain)
    corr_draws = np.asarray([], dtype=float)
    if rank_gain:
        rank_gain_arr = np.asarray(rank_gain)
        error_gain_arr = np.asarray(error_gain)
        corr_draws = []
        for _ in range(int(n_boot)):
            idx = rng.choice(len(rank_gain_arr), len(rank_gain_arr), replace=True)
            corr_draws.append(_spearman(rank_gain_arr[idx], error_gain_arr[idx]))
        corr_draws = np.asarray(corr_draws, dtype=float)
    finite_corr = corr_draws[np.isfinite(corr_draws)]
    performance_link = {
        "graph_ids": used_graphs,
        "structural_rank_gain": rank_gain,
        "absolute_error_gain": error_gain,
        "spearman_rho": rho,
        "ci_low": float(np.quantile(finite_corr, 0.025)) if finite_corr.size else None,
        "ci_high": float(np.quantile(finite_corr, 0.975)) if finite_corr.size else None,
    }

    primary = contrasts["overall"]["structural"]
    specificity = contrasts["overall"]["structural_minus_semantic"]
    if primary["ci_low"] is not None and primary["ci_low"] > 0:
        verdict = "SUPPORTED" if specificity["ci_low"] is not None and specificity["ci_low"] > 0 \
            else "POSITIVE BUT NOT STRUCTURAL-SPECIFIC"
    else:
        verdict = "NO CLEAR SEPARATION"

    return {
        "tasks": tasks,
        "graph_ids": common,
        "bin_labels": list(BIN_LABELS),
        "activity_floor_fraction": float(activity_floor_fraction),
        "n_boot": int(n_boot),
        "summaries": summaries,
        "contrasts": contrasts,
        "bootstrap": bootstrap,
        "performance_link": performance_link,
        "verdict": verdict,
        "criterion": (
            "SUPPORTED iff the paired 95% CI for overall structural effective-rank "
            "(1-hop global RRWP minus 1-hop local RRWP) is above zero and its "
            "structural-minus-semantic specificity CI is above zero."
        ),
    }


def _task_label(task: str) -> str:
    return {
        "zinc": "Dense GRIT\n(global RRWP)",
        "zinc_1hop": "1-hop GRIT\n(global RRWP)",
        "zinc_1hop_local": "1-hop GRIT\n(local RRWP)",
    }.get(task, task)


def _style(task: str) -> dict:
    return {
        "zinc": {"color": "#4c78a8", "marker": "o"},
        "zinc_1hop": {"color": "#f58518", "marker": "s"},
        "zinc_1hop_local": {"color": "#54a24b", "marker": "^"},
    }[task]


def _finite_or_nan(value) -> float:
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.axisbelow": True,
        "figure.dpi": 160,
    })
    return plt


def make_score_scatter(task_results: dict, out: Path) -> str:
    """Deliverable (i): one semantic-vs-structural per-head panel per ZINC model."""
    plt = _setup_matplotlib()
    tasks = list(task_results)
    gsem = np.mean([task_results[t]["score"]["S_sem"].mean() for t in tasks]) + 1e-12
    gstr = np.mean([task_results[t]["score"]["S_str"].mean() for t in tasks]) + 1e-12
    all_xy = []
    for task in tasks:
        score = task_results[task]["score"]
        all_xy.extend((score["S_str"] / gstr).reshape(-1))
        all_xy.extend((score["S_sem"] / gsem).reshape(-1))
    limit = 1.06 * max(max(all_xy), 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), sharex=True, sharey=True,
                             constrained_layout=True)
    scatter = None
    for ax, task in zip(axes, tasks):
        score = task_results[task]["score"]
        L, H = int(score["L"]), int(score["H"])
        x = (score["S_str"] / gstr).reshape(-1)
        y = (score["S_sem"] / gsem).reshape(-1)
        layer = np.repeat(np.arange(L), H)
        ax.plot([0, limit], [0, limit], "k:", lw=1)
        scatter = ax.scatter(x, y, c=layer, cmap="viridis", s=42, alpha=0.88,
                             edgecolors="black", linewidths=0.35, vmin=0, vmax=L - 1)
        ax.set_title(f"{_task_label(task)}\nMAE={score['test_metric']:.4f}", fontweight="bold")
        ax.set_xlim(0, limit); ax.set_ylim(0, limit)
        ax.set_xlabel("structural score  (global-channel normalised)")
    axes[0].set_ylabel("semantic score  (global-channel normalised)")
    fig.colorbar(scatter, ax=axes, label="layer index (0 = input)", shrink=0.82)
    fig.suptitle(
        "BETA · Separate-intervention per-head specialisation on ZINC\n"
        "Same semantic donor swap and mask-frozen structural transposition in every model",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def make_distance_rank_figure(task_results: dict, analysis: dict, out: Path) -> str:
    """Distance-resolved response rank plus the paired global/local contrast."""
    plt = _setup_matplotlib()
    tasks = list(task_results)
    x = np.arange(len(BIN_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), constrained_layout=True)
    for ax, channel in zip(axes[:2], ("semantic", "structural")):
        for task in tasks:
            st = _style(task)
            vals, lo, hi = [], [], []
            for key in BIN_LABELS:
                s = analysis["summaries"][task][channel][key]
                vals.append(_finite_or_nan(s["rank"]))
                lo.append(_finite_or_nan(s["ci_low"]))
                hi.append(_finite_or_nan(s["ci_high"]))
            vals, lo, hi = map(np.asarray, (vals, lo, hi))
            ax.plot(x, vals, marker=st["marker"], color=st["color"], lw=2,
                    label=_task_label(task).replace("\n", " "))
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14)
        ax.set_xticks(x, BIN_LABELS)
        ax.set_xlabel("source–carrier distance [hops]")
        ax.set_ylabel("effective head-response rank")
        ax.set_title(f"{channel.capitalize()} response richness", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    for channel, color, marker in (
        ("structural", "#b22222", "o"),
        ("semantic", "#3366aa", "s"),
    ):
        vals, lo, hi = [], [], []
        for key in BIN_LABELS:
            c = analysis["contrasts"][key][channel]
            vals.append(_finite_or_nan(c["difference"]))
            lo.append(_finite_or_nan(c["ci_low"]))
            hi.append(_finite_or_nan(c["ci_high"]))
        vals, lo, hi = map(lambda z: np.asarray(z, dtype=float), (vals, lo, hi))
        axes[2].errorbar(x, vals, yerr=np.vstack([
                             np.maximum(0.0, vals - lo),
                             np.maximum(0.0, hi - vals),
                         ]),
                         color=color, marker=marker, lw=2, capsize=3,
                         label=channel.capitalize())
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_xticks(x, BIN_LABELS)
    axes[2].set_xlabel("source–carrier distance [hops]")
    axes[2].set_ylabel("rank difference")
    axes[2].set_title("1-hop global RRWP − local RRWP\npaired 95% graph bootstrap CI",
                      fontweight="bold")
    axes[2].legend(frameon=False)
    fig.suptitle(
        "BETA · Reach is amplitude; richness is diversity of amplitude-normalised head patterns\n"
        "Missing points indicate insufficient response above the predeclared activity floor",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def make_verdict_figure(task_results: dict, analysis: dict, out: Path) -> str:
    """Headline overall contrast and graph-level association with the MAE advantage."""
    plt = _setup_matplotlib()
    tasks = list(task_results)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), constrained_layout=True)

    offsets = {"semantic": -0.13, "structural": 0.13}
    colors = {"semantic": "#3366aa", "structural": "#b22222"}
    for channel in ("semantic", "structural"):
        vals, lo, hi = [], [], []
        for task in tasks:
            s = analysis["summaries"][task][channel]["overall"]
            vals.append(_finite_or_nan(s["rank"]))
            lo.append(_finite_or_nan(s["ci_low"]))
            hi.append(_finite_or_nan(s["ci_high"]))
        vals, lo, hi = map(lambda z: np.asarray(z, dtype=float), (vals, lo, hi))
        xpos = np.arange(len(tasks)) + offsets[channel]
        axes[0].errorbar(xpos, vals, yerr=np.vstack([
                             np.maximum(0.0, vals - lo),
                             np.maximum(0.0, hi - vals),
                         ]),
                         fmt="o", ms=7, lw=2, capsize=4, color=colors[channel],
                         label=channel.capitalize())
    axes[0].set_xticks(range(len(tasks)), [_task_label(t) for t in tasks])
    axes[0].set_ylabel("overall effective head-response rank")
    axes[0].set_title("Amplitude-normalised richness", fontweight="bold")
    axes[0].legend(frameon=False)

    contrast_names = ("semantic", "structural", "structural_minus_semantic")
    labels = ("semantic", "structural", "structural − semantic")
    ccolors = ("#3366aa", "#b22222", "#6a3d9a")
    vals, lo, hi = [], [], []
    for name in contrast_names:
        c = analysis["contrasts"]["overall"][name]
        vals.append(_finite_or_nan(c["difference"]))
        lo.append(_finite_or_nan(c["ci_low"]))
        hi.append(_finite_or_nan(c["ci_high"]))
    vals, lo, hi = map(lambda z: np.asarray(z, dtype=float), (vals, lo, hi))
    axes[1].axhline(0, color="black", lw=1)
    for i in range(3):
        axes[1].errorbar(i, vals[i], yerr=[
                             [max(0.0, vals[i] - lo[i])],
                             [max(0.0, hi[i] - vals[i])],
                         ],
                         fmt="o", color=ccolors[i], ms=8, capsize=5, lw=2)
    axes[1].set_xticks(range(3), labels, rotation=12)
    axes[1].set_ylabel("global-RRWP − local-RRWP rank")
    axes[1].set_title("Predeclared paired contrast", fontweight="bold")

    link = analysis["performance_link"]
    dx = np.asarray(link["structural_rank_gain"], dtype=float)
    dy = np.asarray(link["absolute_error_gain"], dtype=float)
    axes[2].axhline(0, color="#777", lw=0.8); axes[2].axvline(0, color="#777", lw=0.8)
    axes[2].scatter(dx, dy, s=28, alpha=0.65, color="#444", edgecolors="white", linewidths=0.3)
    if len(dx) >= 2 and np.std(dx) > 0:
        xx = np.linspace(dx.min(), dx.max(), 100)
        axes[2].plot(xx, np.polyval(np.polyfit(dx, dy, 1), xx), color="#b22222", lw=1.5)
    ci = (link["ci_low"], link["ci_high"])
    axes[2].set_xlabel("per-graph structural-rank gain")
    axes[2].set_ylabel("absolute-error gain\n(local error − global error)")
    axes[2].set_title(
        f"Does richness track performance?\nSpearman ρ={link['spearman_rho']:.2f} "
        f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else
        "Does richness track performance?",
        fontweight="bold",
    )
    fig.suptitle(
        f"BETA VERDICT: {analysis['verdict']}\n"
        "Primary test: global-RRWP > local-RRWP structural rank, beyond semantic differences",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if k != "bootstrap"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _save_raw(task_results: dict, analysis: dict, out_dir: Path) -> tuple[str, str]:
    payload = {}
    for task, item in task_results.items():
        score = item["score"]
        payload[f"{task}__S_sem"] = score["S_sem"]
        payload[f"{task}__S_str"] = score["S_str"]
        collector: ResponseCollector = item["collector"]
        for channel in ("semantic", "structural"):
            records = collector.records[channel]
            payload[f"{task}__{channel}__response"] = np.concatenate(
                [r["response"] for r in records], axis=0
            )
            payload[f"{task}__{channel}__graph_id"] = np.concatenate([
                np.full(len(r["source_nodes"]), r["graph_id"], dtype=np.int64)
                for r in records
            ])
            payload[f"{task}__{channel}__source_node"] = np.concatenate([
                r["source_nodes"] for r in records
            ])
        gids = np.asarray(sorted(collector.abs_error), dtype=np.int64)
        payload[f"{task}__error_graph_id"] = gids
        payload[f"{task}__absolute_error"] = np.asarray(
            [collector.abs_error[int(g)] for g in gids], dtype=float
        )
    npz_path = out_dir / "beta_head_response_richness_zinc.npz"
    np.savez_compressed(npz_path, **payload)
    json_path = out_dir / "beta_head_response_richness_summary.json"
    summary = {
        "status": "BETA",
        "method": (
            "Participation ratio of source-wise L2-normalised per-head functional-carriage "
            "patterns, using the existing separate semantic/structural interventions."
        ),
        "bin_labels": list(BIN_LABELS),
        "analysis": _jsonable(analysis),
        "models": {
            task: {
                "title": item["score"]["title"],
                "test_metric": item["score"]["test_metric"],
                "test_metric_name": item["score"]["test_metric_name"],
                "score_reconstruction_max_abs": item["score_reconstruction_max_abs"],
                "checks": _jsonable(item["score"]["checks"]),
            }
            for task, item in task_results.items()
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(npz_path), str(json_path)


def _mount_drive(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(mount_point, force_remount=False)
    except Exception:  # noqa: BLE001
        log("[drive] google.colab unavailable; using paths as supplied.")


def run(
    *,
    tasks: Sequence[str] = BETA_TASKS,
    out_dir: str = DEFAULT_OUT_DIR,
    ckpt: Optional[dict] = None,
    num_graphs: int = 128,
    donors: int = 8,
    max_sources: Optional[int] = None,
    n_boot: int = 1000,
    activity_floor_fraction: float = 1e-4,
    analysis_seed: int = 0,
    bootstrap_seed: int = 1729,
    partner_match: str = "degree",
    seed: int = 42,
    accelerator: str = "cuda:0",
    num_threads: int = 4,
    mount: bool = True,
    skip_install: bool = False,
    pyg_version: str = "2.2.0",
    force_fresh_grit: bool = False,
) -> dict:
    """Run the isolated three-checkpoint ZINC richness experiment end to end."""
    tasks = tuple(tasks)
    if tasks != BETA_TASKS:
        raise ValueError(
            f"This beta has a predeclared paired design; tasks must be exactly {BETA_TASKS}, "
            f"got {tasks}."
        )
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 <= float(activity_floor_fraction) < 1.0:
        raise ValueError("activity_floor_fraction must lie in [0, 1)")
    if mount:
        _mount_drive()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not skip_install:
        env.install_dependencies(pyg_version=pyg_version)
    else:
        log("[deps] Skipping dependency installation (skip_install=True).")

    task_results = {}
    for task_name in tasks:
        spec: GritTaskSpec = get_task(task_name)
        task_out = output / task_name
        task_out.mkdir(parents=True, exist_ok=True)
        log("\n" + "#" * 88 + f"\n# BETA TASK: {spec.name} ({spec.title})\n" + "#" * 88)

        default_dir = f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"
        repo_dir = Path(spec.grit_repo_dir or default_dir)
        env.clone_grit(
            repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=force_fresh_grit
        )
        for hook in spec.env_hooks:
            hook(repo_dir)
        env.prepare_inprocess_grit(repo_dir)
        config_file = env.resolve_config(spec, repo_dir, task_out)
        chosen_ckpt, _epoch = env.find_checkpoint(
            Path(spec.drive_dir) / "results", (ckpt or {}).get(task_name)
        )
        log(f"[env] {platform.platform()} | python {sys.version.split()[0]}")

        sc = SpecConfig(
            ckpt=str(chosen_ckpt),
            out_dir=str(task_out),
            dataset_dir=str(Path(spec.drive_dir) / "datasets"),
            config_file=config_file,
            accelerator=accelerator,
            seed=seed,
            num_threads=num_threads,
            num_graphs=num_graphs,
            donors=donors,
            analysis_seed=analysis_seed,
            partner_match=partner_match,
        )
        collector = ResponseCollector(task_name)
        score = score_model(
            spec, sc,
            with_attn_routing=False,
            max_sources=max_sources,
            seed=analysis_seed,
            response_observer=collector,
        )
        sem_rebuilt = collector.score_reconstruction("semantic").reshape(score["L"], score["H"])
        str_rebuilt = collector.score_reconstruction("structural").reshape(score["L"], score["H"])
        reconstruction_error = max(
            float(np.max(np.abs(sem_rebuilt - score["S_sem"]))),
            float(np.max(np.abs(str_rebuilt - score["S_str"]))),
        )
        reconstruction_tol = max(
            1e-8,
            1e-4 * float(max(score["S_sem"].max(), score["S_str"].max(), 1e-12)),
        )
        if reconstruction_error > reconstruction_tol:
            raise RuntimeError(
                "Beta response tensor does not reconstruct the production specialisation "
                f"scores: max error={reconstruction_error:.3e} > {reconstruction_tol:.3e}."
            )
        log(f"[beta-audit] distance-resolved responses reconstruct S_sem/S_str: "
            f"max|error|={reconstruction_error:.3e}")
        score.pop("gm", None)
        task_results[task_name] = {
            "score": score,
            "collector": collector,
            "graph_ids": score["graph_ids"],
            "score_reconstruction_max_abs": reconstruction_error,
        }
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    analysis = analyse_response_rank(
        task_results,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
        activity_floor_fraction=activity_floor_fraction,
    )
    figures = {
        "score_scatter": make_score_scatter(
            task_results, output / "fig_beta_zinc_specialisation_scatter.png"
        ),
        "distance_richness": make_distance_rank_figure(
            task_results, analysis, output / "fig_beta_zinc_response_rank_by_distance.png"
        ),
        "verdict": make_verdict_figure(
            task_results, analysis, output / "fig_beta_zinc_response_rank_verdict.png"
        ),
    }
    raw_npz, summary_json = _save_raw(task_results, analysis, output)
    log("\n" + "=" * 88)
    log(f"BETA VERDICT: {analysis['verdict']}")
    log(analysis["criterion"])
    log(f"[done] Outputs: {output}")
    for name, path in figures.items():
        log(f"  {name}: {path}")
    return {
        "status": "BETA",
        "verdict": analysis["verdict"],
        "figures": figures,
        "raw_npz": raw_npz,
        "summary_json": summary_json,
        "analysis": analysis,
        "out_dir": str(output),
    }
