"""Targeted molecular attention context for Chapter 6 head comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _normalise_profile(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    array = np.where(np.isfinite(array), np.maximum(array, 0.0), 0.0)
    total = float(np.sum(array))
    return array / total if total > 1.0e-12 else np.zeros_like(array)


def _head_distance_profiles(
    scores: Mapping[str, Any], head: tuple[int, int]
) -> dict[str, Any]:
    """Extract comparable score and attention distributions for one head."""

    import numpy as np

    from .zinc_cached_rrwp_comparison import group_distance

    layer, index = (int(head[0]), int(head[1]))
    axis = tuple(scores["axis"])
    profiles: dict[str, Any] = {}
    labels: tuple[str, ...] | None = None
    for channel in ("semantic", "structural"):
        exact = scores["channels"][channel]["heatmap_exact_head"]
        if hasattr(exact, "detach"):
            exact = exact.detach().cpu().numpy()
        exact = np.asarray(exact, dtype=np.float64)
        grouped_labels, grouped = group_distance(exact[layer, index], axis)
        labels = grouped_labels if labels is None else labels
        if grouped_labels != labels:
            raise ValueError("semantic and structural distance axes do not agree")
        profiles[channel] = _normalise_profile(grouped)
    attention = scores.get("clean_attention_distance")
    if attention is not None:
        if hasattr(attention, "detach"):
            attention = attention.detach().cpu().numpy()
        attention = np.asarray(attention, dtype=np.float64)
        grouped_labels, grouped = group_distance(attention[layer, index], axis)
        if grouped_labels != labels:
            raise ValueError("attention and score distance axes do not agree")
        profiles["attention"] = _normalise_profile(grouped)
    if labels is None:
        raise ValueError("head score cache has no distance profiles")
    return {"labels": labels, **profiles}


def generate_head_context(
    model_artifacts: Sequence[Mapping[str, Any]],
    representative_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    tasks: Sequence[str] | None = None,
    graph_indices: Sequence[int] = (0, 1),
    accelerator: str = "cuda:0",
    compute_missing: bool = True,
    force: bool = False,
    verbose: bool = True,
    context_name: str = "alignment",
) -> dict[str, Any]:
    """Render selected heads on real molecules using a supplemental exact cache."""

    import matplotlib.pyplot as plt

    from .methodology.grit_figure_data import (
        CHEMISTRY_FOCUS_VERSION,
        SupplementalCache,
        build_verified_grit_figure_runtime,
        collect_attention_examples,
        figure_identity,
        load_canonical_model_record,
        load_canonical_score_artifact,
        methodology_config_from_record,
    )
    from .methodology.grit_figure_plots import (
        plot_attention_grid,
        save_figure_bundle,
    )

    output_dir = Path(output_dir)
    context_name = str(context_name).strip().lower().replace(" ", "_")
    if not context_name or not all(
        character.isalnum() or character in {"_", "-"} for character in context_name
    ):
        raise ValueError(f"invalid head-context name: {context_name!r}")
    cache_name = (
        "alignment-head-attention"
        if context_name == "alignment"
        else f"{context_name}-head-attention"
    )
    diagnostic = (
        "chapter6_alignment_head_context_v1"
        if context_name == "alignment"
        else f"chapter6_{context_name}_head_context_v1"
    )
    figure_dir = output_dir / "figures"
    cache_dir = output_dir / "cache"
    requested = None if tasks is None else {str(task) for task in tasks}
    artifacts = {
        str(record["task"]): record
        for record in model_artifacts
        if requested is None or str(record["task"]) in requested
    }
    rows_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in representative_rows:
        task = str(row["task"])
        if task in artifacts:
            rows_by_task.setdefault(task, []).append(row)

    outputs: list[dict[str, Any]] = []
    warnings: list[str] = []
    task_records: list[dict[str, Any]] = []
    for task, rows in rows_by_task.items():
        artifact_record = artifacts[task]
        score_path = Path(str(artifact_record["score"])).expanduser().resolve()
        artifact_task = str(artifact_record.get("artifact_task") or task)
        model_path = score_path.parents[2] / "model.json"
        protocol_path = score_path.parents[4] / "protocol.json"
        task_record: dict[str, Any] = {
            "task": task,
            "artifact_task": artifact_task,
            "score_path": str(score_path),
            "model_path": str(model_path),
            "protocol_path": str(protocol_path),
            "graph_indices": [int(index) for index in graph_indices],
            "context_name": context_name,
            "status": "started",
        }
        task_records.append(task_record)
        if verbose:
            print(
                f"[head-context:{task}] score={score_path}\n"
                f"[head-context:{task}] model={model_path}\n"
                f"[head-context:{task}] protocol={protocol_path}",
                flush=True,
            )
        try:
            task_record["stage"] = "load canonical score"
            artifact = load_canonical_score_artifact(score_path, expected_task=artifact_task)
            task_record["stage"] = "load canonical model"
            model_record = load_canonical_model_record(model_path, artifact)
            task_record["checkpoint"] = str(model_record.get("checkpoint", ""))
            task_record["checkpoint_exists"] = Path(
                str(model_record.get("checkpoint", ""))
            ).expanduser().is_file()
            task_record["stage"] = "load protocol"
            protocol = methodology_config_from_record(protocol_path, accelerator=accelerator)
            heads = {
                f"comparison_{index}": (int(row["layer"]), int(row["head"]))
                for index, row in enumerate(rows)
            }
            task_record["heads"] = {
                role: list(head) for role, head in heads.items()
            }
            cache = SupplementalCache(cache_dir / task)
            contract = artifact.metadata["contract"]
            cache_contract = {
                "diagnostic": diagnostic,
                "task": artifact_task,
                "canonical_score_sha256": artifact.file_sha256,
                "canonical_contract_fingerprint": artifact.metadata["contract_fingerprint"],
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "heads": {role: list(head) for role, head in heads.items()},
                "graph_indices": [int(index) for index in graph_indices],
                "chemistry_focus_version": CHEMISTRY_FOCUS_VERSION,
            }

            def compute(
                artifact=artifact,
                model_record=model_record,
                protocol=protocol,
                task=task,
                heads=heads,
            ) -> Mapping[str, Any]:
                if verbose:
                    print(
                        f"[head-context:{task}] cache miss; reconstructing model "
                        "and computing two molecule examples",
                        flush=True,
                    )
                runtime = build_verified_grit_figure_runtime(
                    artifact,
                    model_record,
                    protocol,
                    runtime_output_dir=cache_dir / task / "runtime",
                    # Canonical caches from earlier equivalent protocol records
                    # remain scientifically usable here. Checkpoint identity,
                    # model geometry and parameter count are still verified by
                    # the runtime builder; the intervention is rebuilt under
                    # the current adapter.
                    require_protocol_match=False,
                    require_adapter_match=False,
                )
                return collect_attention_examples(
                    runtime,
                    graph_indices=graph_indices,
                    heads=heads,
                )

            if compute_missing:
                task_record["stage"] = "load or compute attention examples"
                payload, cache_path, cache_hit = cache.load_or_compute(
                    cache_name,
                    cache_contract,
                    compute,
                    force=force,
                )
            else:
                task_record["stage"] = "load cached attention examples"
                cached = cache.load(cache_name, cache_contract)
                if cached is None:
                    message = f"{task}: no cached molecular head context"
                    warnings.append(message)
                    task_record.update(status="skipped", error=message)
                    continue
                payload, cache_path = cached
                cache_hit = True
            task_record["attention_cache"] = str(cache_path)
            task_record["cache_hit"] = bool(cache_hit)
            task_record["examples"] = len(payload.get("examples", ()))
            if verbose:
                source = "cache hit" if cache_hit else "computed and cached"
                print(
                    f"[head-context:{task}] {source}: {cache_path}",
                    flush=True,
                )

            row_by_role = {f"comparison_{index}": row for index, row in enumerate(rows)}
            identity = figure_identity(task)
            display_payload = {**dict(payload), **identity}
            task_record["stage"] = "render attention figures"
            for role, head in heads.items():
                row = row_by_role[role]
                role_label = str(row["role"]).capitalize()
                if (
                    "semantic_attention_reach_gap" in row
                    and "structural_attention_reach_gap" in row
                ):
                    title = (
                        f"{role_label}; "
                        f"Δsem={float(row['semantic_attention_reach_gap']):+.2f} hops; "
                        f"Δstr={float(row['structural_attention_reach_gap']):+.2f} hops; "
                        f"max|Δ|={float(row['max_abs_attention_reach_gap']):.2f}"
                    )
                else:
                    title = (
                        f"{role_label}; overlap={float(row['overlap']):.2f}; "
                        f"semantic={float(row['raw_semantic_score']):.3g}; "
                        f"structural={float(row['raw_structural_score']):.3g}"
                    )
                figure = plot_attention_grid(
                    display_payload,
                    role=role,
                    head=head,
                    per_graph_coordinates=None,
                    net_d_rel=float(row["selectivity"]),
                    net_joint_sensitivity=float(row["joint_sensitivity"]),
                    title_label=title,
                    distance_profiles=_head_distance_profiles(artifact.value, head),
                )
                stem = (
                    f"{task}_{context_name}_{role}_layer_{head[0]}_head_{head[1]}"
                )
                optional_metadata = {
                    key: row[key]
                    for key in (
                        "semantic_attention_reach_gap",
                        "structural_attention_reach_gap",
                        "max_abs_attention_reach_gap",
                        "dominant_reach_gap_channel",
                        "dominant_signed_reach_gap",
                    )
                    if key in row
                }
                paths = save_figure_bundle(
                    figure,
                    figure_dir,
                    stem,
                    metadata={
                        "task": task,
                        "context_name": context_name,
                        "artifact_task": artifact_task,
                        "dataset_label": identity["dataset_label"],
                        "model_label": identity["model_label"],
                        "display_title": identity["display_title"],
                        "role": row["role"],
                        "head": list(head),
                        "overlap": float(row["overlap"]),
                        "raw_semantic_score": float(row["raw_semantic_score"]),
                        "raw_structural_score": float(row["raw_structural_score"]),
                        "D_rel": float(row["selectivity"]),
                        "J": float(row["joint_sensitivity"]),
                        "attention_cache": str(cache_path),
                        "cache_hit": bool(cache_hit),
                        "graph_indices": [int(index) for index in graph_indices],
                        **optional_metadata,
                    },
                )
                plt.close(figure)
                outputs.append(
                    {
                        "task": task,
                        "role": row["role"],
                        "layer": head[0],
                        "head": head[1],
                        "model_label": identity["model_label"],
                        "display_title": identity["display_title"],
                        "cache": str(cache_path),
                        **{key: str(path) for key, path in paths.items()},
                    }
                )
            task_record["status"] = "complete"
            task_record["outputs"] = len(heads)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            message = (
                f"{task} [{task_record.get('stage', 'unknown stage')}]: "
                f"{type(error).__name__}: {error}"
            )
            warnings.append(message)
            task_record.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            if verbose:
                print(f"[head-context:error] {message}", flush=True)

    missing_tasks = sorted((requested or set(artifacts)) - set(rows_by_task))
    for task in missing_tasks:
        message = f"{task}: no representative head rows were available"
        warnings.append(message)
        task_records.append({"task": task, "status": "skipped", "error": message})

    summary = {
        "context_name": context_name,
        "requested_tasks": sorted(requested or set(artifacts)),
        "graph_indices": [int(index) for index in graph_indices],
        "outputs": outputs,
        "warnings": warnings,
        "tasks": task_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "head_context_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}


__all__ = ["generate_head_context"]
