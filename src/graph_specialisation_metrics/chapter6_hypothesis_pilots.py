"""Small causal pilots for the Chapter 6 architectural interpretation.

The expensive part of each pilot is deliberately tiny: four held-out graphs,
one source, and one donor by default.  Results are cached per task so rerunning
the Colab normally performs no model forward passes.
"""

from __future__ import annotations

import csv
import gc
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .methodology.causal_spatial_support import Config as CausalConfig
from .methodology.causal_spatial_support import _load_task_context
from .methodology.grit_figure_data import build_verified_grit_figure_runtime
from .methodology.interventions import rrwp_only_donor_swap
from .methodology.runner import _rebuild_graph_events, _stage_plan
from .zinc_cached_rrwp_comparison import TASK_LABELS

PILOT_VERSION = "chapter6-architectural-hypotheses-v1"
CHANNELS = ("semantic", "structural")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_cached_rows(path: Path, contract: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if value.get("contract") != dict(contract) or not isinstance(value.get("rows"), list):
        return None
    return [dict(row) for row in value["rows"]]


def _task_root(roots: Sequence[Path], task: str, seed: int) -> Path | None:
    for root in roots:
        task_dir = Path(root) / task / f"seed_{int(seed)}"
        if (task_dir / "cache/scores/raw.pt").is_file() and (task_dir / "model.json").is_file():
            return Path(root)
    return None


def _selected_events(
    prepared: Any,
    protocol_config: Any,
    plan: Mapping[int, Mapping[str, Any]],
    *,
    graph_id: int,
    channel: str,
    sources_per_graph: int,
    donors_per_source: int,
) -> tuple[list[Any], list[Any]]:
    sources = tuple(
        int(value) for value in plan[int(graph_id)][channel]["sources"][: int(sources_per_graph)]
    )
    variants, records = _rebuild_graph_events(
        prepared,
        protocol_config,
        "causal",
        int(graph_id),
        channel,
        sources,
    )
    keep = [
        index for index, record in enumerate(records) if int(record.draw) < int(donors_per_source)
    ]
    return [variants[index] for index in keep], [records[index] for index in keep]


def _patch_virtual_transport_many(
    backend: Any,
    targets: Sequence[Any],
    donor_transport: Sequence[Any],
    *,
    layer: int,
) -> Any:
    """Patch all heads at the internal VNode transport row in one layer."""

    import torch
    from torch_geometric.data import Batch

    targets = list(targets)
    if not targets:
        raise ValueError("VNode patching requires at least one target")
    if not backend.task.virtual_node:
        raise ValueError("VNode patching requires a registered VNode model")
    if len(donor_transport) != int(backend.gm.L):
        raise ValueError("replacement activations must contain one tensor per layer")
    layer = int(layer)
    if not 0 <= layer < int(backend.gm.L):
        raise IndexError("VNode patch layer is outside model geometry")
    real_rows = sum(int(target.num_nodes) for target in targets)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        routed, edge = output
        donor = donor_transport[layer].to(device=routed.device, dtype=routed.dtype)
        if donor.shape != routed.shape:
            raise RuntimeError(
                f"VNode patch geometry differs: {tuple(donor.shape)} vs {tuple(routed.shape)}"
            )
        rows = torch.arange(
            real_rows,
            real_rows + len(targets),
            device=routed.device,
            dtype=torch.long,
        )
        if int(routed.shape[0]) != real_rows + len(targets):
            raise RuntimeError("VNode rows are not appended after the real-node batch")
        changed = routed.clone()
        changed[rows] = donor[rows]
        return changed, edge

    handle = backend.gm.attn_layers[layer].register_forward_hook(hook)
    try:
        batch = Batch.from_data_list([target.clone() for target in targets]).to(backend.gm.device)
        with torch.no_grad():
            prediction, _target = backend.gm.model(batch)
    finally:
        handle.remove()
    return backend._z(prediction)


def _aligned_fraction(
    clean: np.ndarray, event: np.ndarray, patched: np.ndarray, mode: str
) -> np.ndarray:
    denominator = event - clean
    result = np.full_like(denominator, np.nan, dtype=np.float64)
    valid = np.abs(denominator) > 1.0e-6
    if mode == "restoration":
        numerator = event - patched
    elif mode == "injection":
        numerator = patched - clean
    else:
        raise ValueError(mode)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _measure_vnode(
    runtime: Any,
    protocol_config: Any,
    *,
    graphs: int,
    sources_per_graph: int,
    donors_per_source: int,
) -> list[dict[str, Any]]:
    prepared = runtime.prepared
    plan = _stage_plan(prepared, protocol_config, "causal")
    rows: list[dict[str, Any]] = []
    for graph_id in sorted(plan)[: int(graphs)]:
        base = prepared.grit.eval_ds[int(graph_id)]
        for channel in CHANNELS:
            variants, records = _selected_events(
                prepared,
                protocol_config,
                plan,
                graph_id=int(graph_id),
                channel=channel,
                sources_per_graph=sources_per_graph,
                donors_per_source=donors_per_source,
            )
            if not variants:
                continue
            capture = prepared.backend.capture(
                [base, *variants], require_grad=False, include_virtual_transport=True
            )
            z = capture.z.detach().cpu().numpy().reshape(len(variants) + 1, -1)
            if z.shape[1] != 1:
                raise ValueError("VNode pilot requires one transformed output")
            clean = np.repeat(z[0, 0], len(variants))
            event = z[1:, 0]
            clean_replacements = prepared.backend.replacement_batch(capture, [0] * len(variants))
            event_replacements = prepared.backend.replacement_batch(
                capture, list(range(1, len(variants) + 1))
            )
            for layer in range(int(prepared.grit.L)):
                restored = (
                    _patch_virtual_transport_many(
                        prepared.backend, variants, clean_replacements, layer=layer
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                injected = (
                    _patch_virtual_transport_many(
                        prepared.backend, [base] * len(variants), event_replacements, layer=layer
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                restoration = _aligned_fraction(clean, event, restored, "restoration")
                injection = _aligned_fraction(clean, event, injected, "injection")
                for index, record in enumerate(records):
                    rows.append(
                        {
                            "task": prepared.task.name,
                            "graph": int(graph_id),
                            "source": int(record.source),
                            "donor": int(record.draw),
                            "channel": channel,
                            "layer": layer,
                            "event_effect": float(abs(event[index] - clean[index])),
                            "restoration_fraction": float(restoration[index]),
                            "injection_fraction": float(injection[index]),
                            "matched_mediation_fraction": float(
                                0.5 * (restoration[index] + injection[index])
                            ),
                        }
                    )
    return rows


def _measure_rrwp_reliance(
    runtime: Any,
    protocol_config: Any,
    *,
    graphs: int,
    sources_per_graph: int,
    donors_per_source: int,
) -> list[dict[str, Any]]:
    prepared = runtime.prepared
    plan = _stage_plan(prepared, protocol_config, "causal")
    rows: list[dict[str, Any]] = []
    tolerance = float(protocol_config.numerical.duplicate_tolerance)
    for graph_id in sorted(plan)[: int(graphs)]:
        base = prepared.grit.eval_ds[int(graph_id)]
        full_variants, records = _selected_events(
            prepared,
            protocol_config,
            plan,
            graph_id=int(graph_id),
            channel="structural",
            sources_per_graph=sources_per_graph,
            donors_per_source=donors_per_source,
        )
        if not full_variants:
            continue
        rrwp_variants = [
            rrwp_only_donor_swap(
                base,
                int(record.source),
                int(record.donor_node),
                task=prepared.task,
                duplicate_tolerance=tolerance,
            )
            for record in records
        ]
        capture = prepared.backend.capture(
            [base, *full_variants, *rrwp_variants],
            require_grad=False,
            include_virtual_transport=True,
        )
        z = capture.z.detach().cpu().numpy().reshape(1 + 2 * len(records), -1)
        if z.shape[1] != 1:
            raise ValueError("RRWP pilot requires one transformed output")
        clean = float(z[0, 0])
        full = z[1 : 1 + len(records), 0]
        rrwp = z[1 + len(records) :, 0]
        for index, record in enumerate(records):
            full_effect = float(abs(full[index] - clean))
            rrwp_effect = float(abs(rrwp[index] - clean))
            rows.append(
                {
                    "task": prepared.task.name,
                    "graph": int(graph_id),
                    "source": int(record.source),
                    "donor": int(record.draw),
                    "rrwp_output_effect": rrwp_effect,
                    "full_structural_output_effect": full_effect,
                    "rrwp_fraction_of_full": (
                        rrwp_effect / full_effect if full_effect > 1.0e-6 else float("nan")
                    ),
                }
            )
    return rows


def _bootstrap_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    groups: Sequence[str],
    metric: str,
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[int, list[float]]] = {}
    for row in rows:
        value = float(row[metric])
        if not np.isfinite(value):
            continue
        key = tuple(row[field] for field in groups)
        grouped.setdefault(key, {}).setdefault(int(row["graph"]), []).append(value)
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for key, graph_values in sorted(grouped.items()):
        values = np.asarray(
            [np.mean(graph_values[graph]) for graph in sorted(graph_values)], dtype=np.float64
        )
        point = float(np.mean(values))
        if len(values) > 1 and int(replicates) > 1:
            indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
            draws = np.mean(values[indices], axis=1)
            low, high = np.quantile(draws, (0.025, 0.975))
        else:
            low = high = point
        record = {field: value for field, value in zip(groups, key, strict=True)}
        record.update(
            {
                "metric": metric,
                "mean": point,
                "low": float(low),
                "high": float(high),
                "graphs": len(values),
            }
        )
        output.append(record)
    return output


def _plot_vnode(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [
        task
        for task in ("zinc_1hop_vnode", "zinc_2hop_vnode")
        if any(row["task"] == task for row in rows)
    ]
    if not tasks:
        return []
    summary = _bootstrap_summary(
        rows,
        groups=("task", "channel", "layer"),
        metric="matched_mediation_fraction",
        replicates=500,
        seed=72_031,
    )
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(4.3 * len(tasks), 3.5),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        for channel, colour, marker in (
            ("semantic", "#0072B2", "o"),
            ("structural", "#D55E00", "s"),
        ):
            selected = sorted(
                (row for row in summary if row["task"] == task and row["channel"] == channel),
                key=lambda row: int(row["layer"]),
            )
            if not selected:
                continue
            x = np.asarray([int(row["layer"]) for row in selected])
            mean = np.asarray([float(row["mean"]) for row in selected])
            low = np.asarray([float(row["low"]) for row in selected])
            high = np.asarray([float(row["high"]) for row in selected])
            axis.plot(x, mean, color=colour, marker=marker, label=channel)
            axis.fill_between(x, low, high, color=colour, alpha=0.15)
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_title(TASK_LABELS.get(task, task.replace("_", " ")))
        axis.set_xlabel("patched VNode layer")
        if column == 0:
            axis.set_ylabel("matched mediation fraction")
            axis.legend(frameon=False)
    figure.suptitle("Does the VNode preferentially mediate semantic interventions?")
    return _save_figure(figure, output_dir, "25_vnode_mediation")


def _plot_rrwp(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [
        task
        for task in ("zinc_1hop", "zinc_2hop", "zinc")
        if any(row["task"] == task for row in rows)
    ]
    if not tasks:
        return []
    metrics = ("rrwp_output_effect", "full_structural_output_effect")
    summary = {
        metric: _bootstrap_summary(
            rows,
            groups=("task",),
            metric=metric,
            replicates=500,
            seed=72_047 + index,
        )
        for index, metric in enumerate(metrics)
    }
    lookup = {
        (metric, str(row["task"])): row
        for metric, metric_rows in summary.items()
        for row in metric_rows
    }
    figure, axis = plt.subplots(figsize=(6.0, 3.7), constrained_layout=True)
    x = np.arange(len(tasks), dtype=np.float64)
    width = 0.34
    for offset, metric, label, colour in (
        (-width / 2, metrics[0], "RRWP only", "#0072B2"),
        (width / 2, metrics[1], "full structural swap", "#D55E00"),
    ):
        selected = [lookup[(metric, task)] for task in tasks]
        mean = np.asarray([float(row["mean"]) for row in selected])
        low = np.asarray([float(row["low"]) for row in selected])
        high = np.asarray([float(row["high"]) for row in selected])
        axis.bar(x + offset, mean, width, color=colour, label=label)
        axis.errorbar(
            x + offset,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            color="#222222",
            fmt="none",
            capsize=3,
            linewidth=0.9,
        )
    axis.set_xticks(x, [TASK_LABELS.get(task, task.replace("_", " ")) for task in tasks])
    axis.set_ylabel("absolute output movement")
    axis.set_title("Does dense attention rely more strongly on RRWP?")
    axis.legend(frameon=False)
    return _save_figure(figure, output_dir, "26_rrwp_only_reliance")


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def run(
    roots: Sequence[Path],
    output_dir: Path,
    *,
    seed: int = 42,
    graphs: int = 4,
    sources_per_graph: int = 1,
    donors_per_source: int = 1,
    accelerator: str = "cuda:0",
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run or load the two tiny forward-pass pilots."""

    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    figures_dir = output_dir / "figures"
    warnings: list[str] = []
    all_rows: dict[str, list[dict[str, Any]]] = {"vnode": [], "rrwp": []}
    task_sets = {
        "vnode": ("zinc_1hop_vnode", "zinc_2hop_vnode"),
        "rrwp": ("zinc_1hop", "zinc_2hop", "zinc"),
    }
    for kind, tasks in task_sets.items():
        for task in tasks:
            root = _task_root(roots, task, seed)
            if root is None:
                warnings.append(f"{kind}: {task} has no canonical score/model artifact; skipped")
                continue
            config = CausalConfig(
                canonical_root=root,
                output_dir=output_dir / "runtime",
                tasks=(task,),
                train_seed=int(seed),
                graphs=int(graphs),
                sources_per_graph=int(sources_per_graph),
                donors_per_source=int(donors_per_source),
                accelerator=accelerator,
            )
            try:
                context = _load_task_context(config, task)
                contract = {
                    "version": PILOT_VERSION,
                    "kind": kind,
                    "task": task,
                    "seed": int(seed),
                    "graphs": int(graphs),
                    "sources_per_graph": int(sources_per_graph),
                    "donors_per_source": int(donors_per_source),
                    "score_sha256": context["artifact"].file_sha256,
                }
                cache_path = cache_dir / kind / f"{task}.json"
                rows = None if force else _load_cached_rows(cache_path, contract)
                if rows is None:
                    runtime = build_verified_grit_figure_runtime(
                        context["artifact"],
                        context["model_record"],
                        context["protocol_config"],
                        runtime_output_dir=output_dir / "runtime" / task,
                        require_protocol_match=False,
                    )
                    if kind == "vnode":
                        rows = _measure_vnode(
                            runtime,
                            context["protocol_config"],
                            graphs=graphs,
                            sources_per_graph=sources_per_graph,
                            donors_per_source=donors_per_source,
                        )
                    else:
                        rows = _measure_rrwp_reliance(
                            runtime,
                            context["protocol_config"],
                            graphs=graphs,
                            sources_per_graph=sources_per_graph,
                            donors_per_source=donors_per_source,
                        )
                    _save_json(cache_path, {"contract": contract, "rows": rows})
                    del runtime
                    _release_memory()
                    status = "measured"
                else:
                    status = "cache"
                if verbose:
                    print(f"[chapter6:{kind}] {task}: {status} ({len(rows)} rows)", flush=True)
                all_rows[kind].extend(rows)
            except Exception as error:  # noqa: BLE001 - one pilot must not block the notebook
                warning = f"{kind}: {task}: {type(error).__name__}: {error}"
                warnings.append(warning)
                if verbose:
                    print(f"[chapter6:pilot-warning] {warning}", flush=True)
                _release_memory()

    _write_csv(output_dir / "vnode_mediation_events.csv", all_rows["vnode"])
    _write_csv(output_dir / "rrwp_only_reliance_events.csv", all_rows["rrwp"])
    figures = [
        *_plot_vnode(all_rows["vnode"], figures_dir),
        *_plot_rrwp(all_rows["rrwp"], figures_dir),
    ]
    summary = {
        "version": PILOT_VERSION,
        "settings": {
            "seed": int(seed),
            "graphs": int(graphs),
            "sources_per_graph": int(sources_per_graph),
            "donors_per_source": int(donors_per_source),
        },
        "warnings": warnings,
        "figures": [str(path) for path in figures],
        "estimands": {
            "vnode": (
                "symmetric injection/restoration fraction after replacing all head-output "
                "channels at the internal VNode row in one layer"
            ),
            "rrwp": (
                "absolute transformed-output response to an RRWP-only donor footprint swap, "
                "paired with the canonical complete structural swap"
            ),
        },
    }
    _save_json(output_dir / "summary.json", summary)
    return {**summary, "vnode_rows": all_rows["vnode"], "rrwp_rows": all_rows["rrwp"]}


__all__ = ["PILOT_VERSION", "run"]
