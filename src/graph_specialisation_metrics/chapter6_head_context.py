"""Targeted molecular attention context for Chapter 6 alignment comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


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
) -> dict[str, Any]:
    """Render a small matched-head comparison using real molecular attention.

    Each task contributes the lower-alignment active head and its sensitivity-
    matched higher-alignment comparison selected by the spatial explorer. The
    supplemental attention sweep is cached under the canonical score identity.
    """

    import matplotlib.pyplot as plt

    from .methodology.grit_figure_data import (
        CHEMISTRY_FOCUS_VERSION,
        SupplementalCache,
        build_verified_grit_figure_runtime,
        collect_attention_examples,
        load_canonical_model_record,
        load_canonical_score_artifact,
        methodology_config_from_record,
    )
    from .methodology.grit_figure_plots import (
        plot_attention_grid,
        save_figure_bundle,
    )

    output_dir = Path(output_dir)
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
                "diagnostic": "chapter6_alignment_head_context_v1",
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
                    # adapter version, model geometry, and parameter count are
                    # still verified by the runtime builder.
                    require_protocol_match=False,
                )
                return collect_attention_examples(
                    runtime,
                    graph_indices=graph_indices,
                    heads=heads,
                )

            if compute_missing:
                task_record["stage"] = "load or compute attention examples"
                payload, cache_path, cache_hit = cache.load_or_compute(
                    "alignment-head-attention",
                    cache_contract,
                    compute,
                    force=force,
                )
            else:
                task_record["stage"] = "load cached attention examples"
                cached = cache.load("alignment-head-attention", cache_contract)
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
            task_record["stage"] = "render attention figures"
            for role, head in heads.items():
                row = row_by_role[role]
                role_label = str(row["role"]).capitalize()
                title = (
                    f"{role_label}; overlap={float(row['overlap']):.2f}; "
                    f"semantic={float(row['raw_semantic_score']):.3g}; "
                    f"structural={float(row['raw_structural_score']):.3g}"
                )
                figure = plot_attention_grid(
                    payload,
                    role=role,
                    head=head,
                    per_graph_coordinates=None,
                    net_d_rel=float(row["selectivity"]),
                    net_joint_sensitivity=float(row["joint_sensitivity"]),
                    title_label=title,
                )
                stem = f"{task}_{role}_layer_{head[0]}_head_{head[1]}"
                paths = save_figure_bundle(
                    figure,
                    figure_dir,
                    stem,
                    metadata={
                        "task": task,
                        "artifact_task": artifact_task,
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
                    },
                )
                plt.close(figure)
                outputs.append(
                    {
                        "task": task,
                        "role": row["role"],
                        "layer": head[0],
                        "head": head[1],
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
