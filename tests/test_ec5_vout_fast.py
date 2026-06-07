import importlib.util
import sys
from pathlib import Path

import torch


def load_runner():
    path = Path(__file__).resolve().parents[1] / "experiments/graphbench/training/ec5_vout_fast.py"
    spec = importlib.util.spec_from_file_location("ec5_vout_fast", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ec5_vout_fast"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_record(num_nodes=8):
    node_features = torch.eye(9)[:num_nodes].int().tolist()
    undirected = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 0),
        (0, 4),
        (2, 6),
    ]
    edges = undirected + [(v, u) for u, v in undirected]
    return {
        "node_features": node_features,
        "edge_index": [[u for u, _ in edges], [v for _, v in edges]],
        "duty": 0.5,
        "vout": 30.0,
    }


def test_ec5_json_preprocessing_and_collate():
    ec5 = load_runner()
    graph = ec5.graph_from_json_record(fake_record())
    assert graph.num_nodes == 8
    assert graph.edge_index.shape == (2, 20)
    assert graph.node_type.tolist() == list(range(8))
    assert torch.isclose(graph.y, torch.tensor(0.55))

    batch = ec5.collate_ec_graphs([graph, graph])
    assert batch["node_type"].shape == (2, 8)
    assert batch["edge_input"].shape == (2, 8, 8, 10)
    assert batch["rwse"].shape == (2, 8, 10)
    assert batch["rrwp"].shape == (2, 8, 8, 11)
    assert batch["duty"].shape == (2, 1)
    assert graph.cache is not None


def test_model_sizes_and_forward_smoke():
    ec5 = load_runner()
    cfg = ec5.EC5FastConfig()
    graphs = [ec5.graph_from_json_record(fake_record()) for _ in range(4)]
    batch = ec5.collate_ec_graphs(graphs)

    for model_name in ("graphormer", "graphgps", "grit"):
        model = ec5.build_model(model_name, cfg)
        n_params = ec5.count_parameters(model)
        lo, hi = ec5.DEFAULT_SIZE_BANDS[model_name]
        assert lo <= n_params <= hi
        model.eval()
        with torch.no_grad():
            pred = model(batch)
        assert pred.shape == (4,)
        assert torch.isfinite(pred).all()


def test_one_optimizer_step_all_models():
    ec5 = load_runner()
    cfg = ec5.EC5FastConfig()
    graphs = [ec5.graph_from_json_record(fake_record()) for _ in range(8)]
    batch = ec5.collate_ec_graphs(graphs)

    for model_name in ("graphormer", "graphgps", "grit"):
        model = ec5.build_model(model_name, cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        pred = model(batch)
        loss = torch.nn.functional.mse_loss(pred, batch["y"])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
