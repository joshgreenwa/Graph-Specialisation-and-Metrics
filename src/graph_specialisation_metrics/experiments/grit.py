"""GRIT experiments on ZINC and QM9."""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from graph_specialisation_metrics.distance import shortest_path_distances
from graph_specialisation_metrics.interventions import (
    semantic_donor_swap,
    structural_donor_swap,
)
from graph_specialisation_metrics.runner import (
    align_distance_contributions,
    clean_output_gradients,
    compute_channel_score,
    distance_categories,
    mean_channel_scores,
    training_target_scale,
)
from graph_specialisation_metrics.sampling import (
    SemanticDonorPool,
    analysis_indices,
    sample_structural_donors,
)

from . import ExperimentSetupError, dataset_lock, save_scores

UPSTREAM_URL = "https://github.com/LiamMa/GRIT.git"
UPSTREAM_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
VARIANTS = ("dense", "1hop", "1hop_vnode", "2hop", "2hop_vnode")


def jobs(config: dict[str, Any], *, fast: bool = False) -> list[dict[str, Any]]:
    """Return the configured dataset, variant, and seed jobs."""

    dataset = str(config.get("data", {}).get("name", "")).lower()
    if dataset not in {"zinc", "qm9"}:
        raise ValueError("GRIT data.name must be 'zinc' or 'qm9'")
    variants = list(config.get("model", {}).get("variants", VARIANTS))
    if unknown := sorted(set(variants) - set(VARIANTS)):
        raise ValueError(f"unknown GRIT variants: {unknown}")
    seeds = [int(seed) for seed in config.get("training", {}).get("seeds", (0, 1, 2))]
    if not seeds or not variants:
        raise ValueError("at least one GRIT variant and seed are required")
    table = [
        {"experiment": "grit", "dataset": dataset, "variant": variant, "seed": seed}
        for variant in variants
        for seed in seeds
    ]
    job_index = config.get("job_index")
    if job_index is not None:
        index = int(job_index)
        if index < 0 or index >= len(table):
            raise IndexError(f"job_index {index} is outside 0..{len(table) - 1}")
        return [table[index]]
    return table[:1] if fast else table


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise ExperimentSetupError(f"required executable is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ExperimentSetupError(
            f"command failed with status {exc.returncode}: {' '.join(command)}"
        ) from exc


def _checkout(root: Path) -> Path:
    """Check out and patch the required GRIT commit."""

    repo = root / "upstream"
    if not (repo / ".git").is_dir():
        if repo.exists():
            raise ExperimentSetupError(f"{repo} exists but is not a Git checkout")
        root.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", UPSTREAM_URL, str(repo)])
    _run(["git", "fetch", "--quiet", "origin", UPSTREAM_COMMIT], cwd=repo)
    _run(["git", "checkout", "--quiet", "--detach", UPSTREAM_COMMIT], cwd=repo)
    _run(["git", "reset", "--hard", "--quiet", UPSTREAM_COMMIT], cwd=repo)
    _run(["git", "clean", "-fd", "--quiet"], cwd=repo)

    assets = files(__package__)
    for name in ("grit_upstream.patch", "grit_compat.patch", "grit_qm9.patch"):
        patch = Path(str(assets.joinpath(name)))
        _run(["git", "apply", "--ignore-space-change", "--check", str(patch)], cwd=repo)
        _run(["git", "apply", "--ignore-space-change", str(patch)], cwd=repo)
    return repo


def _require_runtime() -> None:
    required = (
        "ogb",
        "sklearn",
        "torch_geometric",
        "torchmetrics",
        "yacs",
    )
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ExperimentSetupError(
            "GRIT runtime dependencies are missing: "
            + ", ".join(missing)
            + ". Install this project with its GRIT extra before training."
        )


def _base_config(repo: Path, dataset: str) -> dict[str, Any]:
    if dataset == "zinc":
        path = repo / "configs/GRIT/zinc-GRIT-RRWP.yaml"
    else:
        path = Path(str(files(__package__).joinpath("grit_qm9.yaml")))
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"invalid GRIT config: {path}")
    return result


def _dataset_root(config: dict[str, Any], output_dir: Path) -> Path:
    configured = config.get("data", {}).get("root")
    root = (
        Path(str(configured)).expanduser()
        if configured
        else Path(output_dir).resolve().parent / "data"
    )
    return root.resolve()


def _resolved_config(
    repo: Path,
    config: dict[str, Any],
    job: dict[str, Any],
    run_dir: Path,
    data_root: Path,
    *,
    fast: bool,
) -> Path:
    """Write the GraphGym config for one run."""

    dataset, variant = job["dataset"], job["variant"]
    resolved = _base_config(repo, dataset)
    model = config.get("model", {})
    training = config.get("training", {})
    layers = int(model.get("layers", 10))
    heads = int(model.get("heads", 8))
    hidden = int(model.get("hidden_dim", 64))
    rrwp = int(model.get("rrwp_steps", 21))
    if (layers, heads, hidden, rrwp) != (10, 8, 64, 21):
        raise ValueError("GRIT configs require 10 layers, 8 heads, width 64 and RRWP-21")

    # Keep each GraphGym run in its own output directory.
    resolved["out_dir"] = str(run_dir)
    resolved["tensorboard_each_run"] = False
    resolved["tensorboard_agg"] = False
    resolved.setdefault("wandb", {})["use"] = False
    resolved.setdefault("dataset", {})["dir"] = str(data_root)
    resolved.setdefault("posenc_RRWP", {})["ksteps"] = rrwp
    resolved.setdefault("gt", {}).update(layers=layers, n_heads=heads, dim_hidden=hidden)
    resolved.setdefault("gnn", {})["dim_inner"] = hidden
    resolved.setdefault("train", {}).update(
        batch_size=int(training.get("batch_size", 32 if dataset == "zinc" else 128)),
        enable_ckpt=True,
        ckpt_best=True,
        ckpt_clean=True,
        auto_resume=False,
    )
    epochs = int(training.get("epochs", 2000 if dataset == "zinc" else 300))
    warmup = int(training.get("warmup_epochs", 50 if dataset == "zinc" else 10))
    if fast:
        epochs, warmup = 1, 0
        resolved["train"]["batch_size"] = min(int(resolved["train"]["batch_size"]), 16)
    resolved.setdefault("optim", {}).update(
        max_epoch=epochs,
        num_warmup_epochs=warmup,
        base_lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    attention = resolved["gt"].setdefault("attn", {})
    attention.update(
        full_attn=variant == "dense",
        sparsity="full" if variant == "dense" else "k_hop",
        hops=2 if variant.startswith("2hop") else 1,
        global_vnode=variant.endswith("vnode"),
    )
    path = run_dir / "config.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return path


def _checkpoint(run_dir: Path, *, config_mtime: float) -> Path:
    candidates = [
        path for path in run_dir.glob("**/ckpt/*.ckpt") if path.stat().st_mtime >= config_mtime
    ]
    if not candidates:
        raise ExperimentSetupError(f"upstream GRIT completed without a checkpoint under {run_dir}")
    numeric = [path for path in candidates if path.stem.isdigit()]
    return (
        max(numeric, key=lambda path: int(path.stem))
        if numeric
        else max(candidates, key=lambda path: path.stat().st_mtime)
    )


def train(config: dict[str, Any], *, output_dir: Path, fast: bool = False) -> dict[str, Any]:
    """Train the selected GRIT jobs."""

    output_dir = Path(output_dir).expanduser().resolve()
    table = jobs(config, fast=fast)
    _require_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = _dataset_root(config, output_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    repo = _checkout(output_dir / "grit_upstream")
    checkpoints: list[Path] = []
    for job in table:
        run_dir = output_dir / job["dataset"] / job["variant"] / f"seed_{job['seed']}"
        config_file = _resolved_config(repo, config, job, run_dir, data_root, fast=fast)
        config_mtime = config_file.stat().st_mtime
        command = [
            sys.executable,
            "main.py",
            "--cfg",
            str(config_file),
            "--repeat",
            "1",
            "seed",
            str(job["seed"]),
            "accelerator",
            str(config.get("training", {}).get("device", "cuda:0")),
            "num_threads",
            str(int(config.get("training", {}).get("num_threads", 4))),
        ]
        _run(command, cwd=repo)
        checkpoints.append(_checkpoint(run_dir, config_mtime=config_mtime))
    return {"checkpoints": checkpoints, "jobs": table}


def _one_checkpoint(checkpoint: Path | None, output_dir: Path) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path(output_dir).glob("**/ckpt/*.ckpt"))
    if len(candidates) != 1:
        raise ExperimentSetupError(
            f"GRIT scoring requires exactly one checkpoint; found {len(candidates)} under "
            f"{output_dir}. Pass --checkpoint explicitly."
        )
    return candidates[0].resolve()


def _checkpoint_config(checkpoint: Path) -> Path:
    for parent in (checkpoint.parent, *checkpoint.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    raise ExperimentSetupError(f"no resolved config.yaml found above {checkpoint}")


def _load_upstream(checkout: Path, config_file: Path, config: dict[str, Any]):
    """Load the GraphGym model and data loaders."""

    checkout_string = str(checkout)
    if checkout_string not in sys.path:
        sys.path.insert(0, checkout_string)
    loaded = sys.modules.get("grit")
    if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(checkout_string):
        for name in [name for name in sys.modules if name == "grit" or name.startswith("grit.")]:
            del sys.modules[name]
    importlib.import_module("grit")

    import torch
    from torch_geometric.graphgym.config import assert_cfg, cfg, set_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model

    cfg.defrost()
    cfg.clear()
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(config_file))
    assert_cfg(cfg)
    requested = str(
        config.get("scoring", {}).get("device", config.get("training", {}).get("device", "cuda:0"))
    )
    device = torch.device(
        requested if not requested.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    cfg.accelerator = str(device)
    cfg.device = str(device)
    dataset_name = str(config.get("data", {}).get("name", "grit"))
    with dataset_lock(Path(str(cfg.dataset.dir)), dataset_name):
        loaders = create_loader()
    model = create_model(to_device=False)
    return cfg, loaders, model, device


def _load_weights(model: Any, checkpoint: Path) -> None:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ExperimentSetupError(f"invalid GraphGym checkpoint: {checkpoint}")
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise ExperimentSetupError(f"checkpoint has no model state: {checkpoint}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ExperimentSetupError(
            f"checkpoint does not match its resolved GRIT config: {checkpoint}\n{exc}"
        ) from exc


def _attention_layers(model: Any) -> list[Any]:
    network = getattr(model, "model", model)
    layers = list(getattr(network, "layers", ()))
    if not layers or any(not hasattr(layer, "attention") for layer in layers):
        raise ExperimentSetupError("patched GRIT model does not expose its attention layers")
    return layers


def _forward_head_outputs(model: Any, data: Any, device: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return the prediction and each layer's recorded head output."""

    from torch_geometric.data import Batch

    head_outputs: list[Any] = []
    handles = [
        layer.attention.register_forward_hook(
            lambda _module, _inputs, output, rows=head_outputs: rows.append(output[0])
        )
        for layer in _attention_layers(model)
    ]
    try:
        prediction, _target = model(Batch.from_data_list([data.clone()]).to(device))
    finally:
        for handle in handles:
            handle.remove()
    if len(head_outputs) != len(handles):
        raise RuntimeError("GRIT did not record every layer's head output")
    shapes = {tuple(value.shape) for value in head_outputs}
    if len(shapes) != 1 or next(iter(shapes))[1] <= 0:
        raise RuntimeError(f"inconsistent GRIT head-output shapes: {sorted(shapes)}")
    return prediction, tuple(head_outputs)


def _training_scale(loader: Any) -> np.ndarray:
    targets = [np.asarray(data.y.detach().cpu()).reshape(-1) for data in loader.dataset]
    if not targets:
        raise ExperimentSetupError("GRIT training split is empty")
    return training_target_scale(np.stack(targets, axis=0))


def _dense_rrwp(data: Any) -> tuple[Any, Any]:
    import torch

    node_count = int(data.num_nodes)
    width = int(data.rrwp_val.shape[-1])
    values = data.rrwp_val.new_zeros((node_count, node_count, width))
    present = torch.zeros((node_count, node_count), dtype=torch.bool, device=values.device)
    left, right = data.rrwp_index
    values[left, right] = data.rrwp_val
    present[left, right] = True
    return values, present


def _structural_profiles(data: Any) -> list[dict[str, Any]]:
    values, present = _dense_rrwp(data)
    node_fields = (data.rrwp, data.deg.reshape(-1, 1), data.log_deg.reshape(-1, 1))
    result = []
    for node in range(int(data.num_nodes)):
        result.append(
            {
                "node": tuple(value[node].detach().cpu().numpy() for value in node_fields),
                "row": values[node].detach().cpu().numpy(),
                "column": values[:, node].detach().cpu().numpy(),
                "row_present": present[node].detach().cpu().numpy(),
                "column_present": present[:, node].detach().cpu().numpy(),
            }
        )
    return result


def _structural_donor_swap(data: Any, source: int, donor: int) -> Any:
    """Apply a structural donor-swap without changing connectivity or attention support."""

    import torch

    donor_swap = data.clone()
    values, present = _dense_rrwp(data)
    nodes, pairs = structural_donor_swap(
        {"rrwp": data.rrwp, "deg": data.deg, "log_deg": data.log_deg},
        {"rrwp": values, "present": present},
        source,
        donor,
    )
    donor_swap.rrwp = nodes["rrwp"]
    donor_swap.deg = nodes["deg"]
    donor_swap.log_deg = nodes["log_deg"]
    left, right = torch.nonzero(pairs["present"], as_tuple=True)
    donor_swap.rrwp_index = torch.stack((left, right), dim=0)
    donor_swap.rrwp_val = pairs["rrwp"][left, right]
    return donor_swap


def _semantic_pool(graphs: Sequence[tuple[Any, Any]]) -> SemanticDonorPool:
    attributes = {graph_id: data.x.detach().cpu().numpy() for graph_id, data in graphs}
    degrees = {
        graph_id: data.deg.detach().cpu().numpy().astype(np.int64, copy=False)
        for graph_id, data in graphs
    }
    return SemanticDonorPool(attributes, degrees)


def _donor_swap_head_outputs(model: Any, donor_swaps: Iterable[Any], device: Any) -> Any:
    import torch

    rows = []
    with torch.no_grad():
        for donor_swap in donor_swaps:
            _prediction, head_outputs = _forward_head_outputs(model, donor_swap, device)
            rows.append(torch.stack(head_outputs, dim=0))
    if not rows:
        raise ValueError("no donor-swaps were generated")
    return torch.stack(rows, dim=0)


def _score_graph(
    model: Any,
    data: Any,
    graph_id: int,
    pool: SemanticDonorPool,
    scale: np.ndarray,
    *,
    sources: int,
    donor_swaps_per_source: int,
    rng: np.random.Generator,
    device: Any,
    vnode: bool,
) -> tuple[Any, Any] | None:
    import torch

    structural_profiles = _structural_profiles(data)
    candidates = rng.permutation(int(data.num_nodes))
    selected: list[tuple[int, np.ndarray]] = []
    for source_value in candidates:
        source = int(source_value)
        structural = sample_structural_donors(
            structural_profiles, source, donor_swaps_per_source, rng
        )
        if len(structural) == 0:
            continue
        eligible = pool.eligible(
            data.x[source].detach().cpu().numpy(),
            int(data.deg[source]),
        )
        if eligible:
            selected.append((source, structural))
        if len(selected) == sources:
            break
    if not selected:
        return None

    clean_prediction, clean_tuple = _forward_head_outputs(model, data, device)
    clean_gradients = clean_output_gradients(clean_prediction, clean_tuple, scale).detach()
    clean = torch.stack(clean_tuple, dim=0).detach()
    all_distances = shortest_path_distances(data.edge_index, int(data.num_nodes))

    semantic_donor_swaps, structural_donor_swaps = [], []
    semantic_sources, structural_sources = [], []
    semantic_distances, structural_distances = [], []
    for source, structural_donors in selected:
        semantic_donors = pool.sample(
            data.x[source].detach().cpu().numpy(),
            int(data.deg[source]),
            donor_swaps_per_source,
            rng,
        )
        distance = all_distances[source].astype(object)
        if vnode:
            distance = np.concatenate((distance, np.asarray(["virtual-node"], dtype=object)))
        for donor in semantic_donors:
            donor_swap = data.clone()
            donor_swap.x = semantic_donor_swap(donor_swap.x, source, donor.attributes)
            semantic_donor_swaps.append(donor_swap)
            semantic_sources.append(source)
            semantic_distances.append(distance)
        for donor in structural_donors:
            structural_donor_swaps.append(_structural_donor_swap(data, source, int(donor)))
            structural_sources.append(source)
            structural_distances.append(distance)

    semantic = compute_channel_score(
        clean,
        _donor_swap_head_outputs(model, semantic_donor_swaps, device),
        clean_gradients,
        [graph_id] * len(semantic_donor_swaps),
        semantic_sources,
        head_output_row_distances=np.stack(semantic_distances),
    )
    structural = compute_channel_score(
        clean,
        _donor_swap_head_outputs(model, structural_donor_swaps, device),
        clean_gradients,
        [graph_id] * len(structural_donor_swaps),
        structural_sources,
        head_output_row_distances=np.stack(structural_distances),
    )
    return semantic, structural


def _variant_name(cfg: Any) -> str:
    if bool(cfg.gt.attn.get("full_attn", False)):
        return "Dense GRIT"
    result = f"{int(cfg.gt.attn.hops)}-hop"
    return result + (" + VNode" if bool(cfg.gt.attn.get("global_vnode", False)) else "")


def score(
    config: dict[str, Any],
    *,
    checkpoint: Path | None,
    output_dir: Path,
    fast: bool = False,
) -> Path:
    """Compute specialisation scores and distance-resolved score contributions."""

    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = _one_checkpoint(checkpoint, output_dir)
    _require_runtime()
    checkout = _checkout(output_dir / "grit_upstream")
    cfg, loaders, model, device = _load_upstream(
        checkout, _checkpoint_config(checkpoint_path), config
    )
    _load_weights(model, checkpoint_path)
    model.to(device).eval()
    if len(loaders) < 3:
        raise ExperimentSetupError("GRIT dataset did not provide train/validation/test splits")
    scale = _training_scale(loaders[0])

    settings = config.get("scoring", {})
    graph_count = int(settings.get("graphs", 48))
    source_count = int(settings.get("sources_per_graph", 6))
    donor_swaps_per_source = int(settings.get("donor_swaps_per_source", 8))
    semantic_pool_graphs = int(settings.get("semantic_pool_graphs", 2000))
    if fast:
        graph_count = source_count = donor_swaps_per_source = 1
        semantic_pool_graphs = min(semantic_pool_graphs, 8)
    if min(graph_count, source_count, donor_swaps_per_source, semantic_pool_graphs) <= 0:
        raise ValueError("GRIT scoring counts must all be positive")

    train_dataset = loaders[0].dataset
    test_dataset = loaders[2].dataset
    available = len(test_dataset)
    if available < 1:
        raise ExperimentSetupError("GRIT test split is empty")
    analysis_seed = int(settings.get("seed", 31415))
    try:
        graph_ids, semantic_pool_ids = analysis_indices(
            available,
            graph_count,
            len(train_dataset),
            semantic_pool_graphs,
            analysis_seed,
        )
    except ValueError as exc:
        raise ExperimentSetupError(str(exc)) from exc
    semantic_pool = [
        (("train", int(index)), train_dataset[int(index)].clone()) for index in semantic_pool_ids
    ]
    pool = _semantic_pool(semantic_pool)
    rng = np.random.default_rng(analysis_seed + 2)
    semantic_results, structural_results = [], []
    for graph_id in graph_ids:
        channel_scores = _score_graph(
            model,
            test_dataset[int(graph_id)].clone(),
            int(graph_id),
            pool,
            scale,
            sources=source_count,
            donor_swaps_per_source=donor_swaps_per_source,
            rng=rng,
            device=device,
            vnode=bool(cfg.gt.attn.get("global_vnode", False)),
        )
        if channel_scores is not None:
            semantic, structural = channel_scores
            semantic_results.append(semantic)
            structural_results.append(structural)
    if not semantic_results:
        raise ExperimentSetupError("no test graph had eligible semantic and structural donor-swaps")
    if len(semantic_results) < graph_count:
        raise ExperimentSetupError(
            f"only {len(semantic_results)} of {graph_count} requested test graphs had "
            "eligible donor-swaps"
        )

    semantic = mean_channel_scores(semantic_results)
    structural = mean_channel_scores(structural_results)
    categories = distance_categories((semantic, structural))
    layer_count, head_count = semantic.score.shape
    size = layer_count * head_count
    return save_scores(
        output_dir,
        semantic.score.reshape(-1),
        structural.score.reshape(-1),
        semantic_distance_contributions=align_distance_contributions(semantic, categories).reshape(
            size, -1
        ),
        structural_distance_contributions=align_distance_contributions(
            structural, categories
        ).reshape(size, -1),
        distance_categories=categories,
        dataset=np.repeat(str(cfg.dataset.name), size),
        variant=np.repeat(_variant_name(cfg), size),
        seed=np.repeat(int(cfg.seed), size),
        layer=np.repeat(np.arange(layer_count), head_count),
        head=np.tile(np.arange(head_count), layer_count),
    )
