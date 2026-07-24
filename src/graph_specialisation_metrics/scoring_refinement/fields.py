"""Official-GRIT attention, message, and routed-output collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class HeadFields:
    """Dense fields for one GRIT layer on one graph.

    ``attention`` is ``[H,N,N]`` in query-by-key coordinates, ``message`` is
    ``[H,N,N,D]``, ``mask`` is ``[H,N,N]``, and ``routed_output`` is
    ``[N,H,D]``.
    """

    layer: int
    attention: Any
    message: Any
    mask: Any
    routed_output: Any


@dataclass
class GraphCapture:
    prediction: Any
    target: Any
    layers: list[HeadFields]


class GritFieldCollector:
    """Collect complete sender-indexed fields from ``GritHeadModel``.

    The message definition exactly mirrors the official GRIT attention module:
    ``V_h[src] + wE @ VeRow`` when edge enhancement is enabled.
    """

    def __init__(self, grit_model: Any) -> None:
        self.gm = grit_model

    def collect(self, data: Any, *, require_grad: bool = False) -> GraphCapture:
        return self.collect_many([data], require_grad=require_grad)[0]

    def collect_many(
        self,
        data_list: Sequence[Any],
        *,
        require_grad: bool = False,
    ) -> list[GraphCapture]:
        """Collect a same-purpose event group in one batched GRIT forward."""

        import torch
        from torch_geometric.data import Batch

        if not data_list:
            return []
        if require_grad and len(data_list) != 1:
            raise ValueError("gradient collection is defined for one clean graph at a time")
        counts = [int(data.num_nodes) for data in data_list]
        offsets = np.cumsum([0, *counts]).tolist()
        batch = Batch.from_data_list([data.clone() for data in data_list]).to(self.gm.device)
        records: list[list[HeadFields | None]] = [
            [None] * len(data_list) for _ in range(int(self.gm.L))
        ]
        layer_by_module = {
            id(module): layer for layer, module in enumerate(self.gm.attn_layers)
        }

        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            layer = layer_by_module[id(module)]
            pyg_batch = inputs[0]
            routed = output[0] if isinstance(output, (tuple, list)) else output
            edge_index = pyg_batch.edge_index.long()
            key = edge_index[0]
            query = edge_index[1]
            attention_e = pyg_batch.attn.squeeze(-1).float()
            heads = int(attention_e.shape[1])
            message_e = pyg_batch.V_h[key].float()
            edge_state = getattr(pyg_batch, "wE", None)
            if getattr(module, "edge_enhance", False) and edge_state is not None:
                edge_state = edge_state.reshape(-1, heads, message_e.shape[-1]).float()
                message_e = message_e + torch.einsum(
                    "ehd,dhc->ehc", edge_state, module.VeRow.float()
                )
            dim = int(message_e.shape[-1])
            for graph_index, n in enumerate(counts):
                start, stop = offsets[graph_index], offsets[graph_index + 1]
                attention = torch.zeros(
                    heads, n, n, device=attention_e.device, dtype=torch.float32
                )
                message = torch.zeros(
                    heads, n, n, dim, device=message_e.device, dtype=torch.float32
                )
                mask = torch.zeros(
                    heads, n, n, device=attention_e.device, dtype=torch.bool
                )
                in_graph = (
                    (key >= start)
                    & (key < stop)
                    & (query >= start)
                    & (query < stop)
                )
                if bool(in_graph.any()):
                    local_key = key[in_graph] - start
                    local_query = query[in_graph] - start
                    attention[:, local_query, local_key] = attention_e[
                        in_graph
                    ].transpose(0, 1)
                    message[:, local_query, local_key] = message_e[in_graph].permute(
                        1, 0, 2
                    )
                    mask[:, local_query, local_key] = True
                output = routed[start:stop]
                records[layer][graph_index] = HeadFields(
                    layer=layer,
                    attention=attention.detach(),
                    message=message.detach(),
                    mask=mask.detach(),
                    routed_output=output if require_grad else output.detach(),
                )

        handles = [module.register_forward_hook(hook) for module in self.gm.attn_layers]
        try:
            context = torch.enable_grad() if require_grad else torch.no_grad()
            with context:
                prediction, target = self.gm.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        if any(item is None for layer in records for item in layer):
            missing = [
                (layer, graph)
                for layer, values in enumerate(records)
                for graph, item in enumerate(values)
                if item is None
            ]
            raise RuntimeError(f"GRIT field hooks did not fire for layer/graphs {missing}")
        captures = []
        for graph_index in range(len(data_list)):
            captures.append(
                GraphCapture(
                    prediction=prediction[graph_index:graph_index + 1],
                    target=target[graph_index:graph_index + 1],
                    layers=[
                        records[layer][graph_index]
                        for layer in range(int(self.gm.L))
                        if records[layer][graph_index] is not None
                    ],
                )
            )
        return captures

    def clean_with_gradients(self, data: Any) -> tuple[GraphCapture, list[Any]]:
        """Capture clean fields and ``d prediction_t / d wV_l`` for every output."""

        import torch

        capture = self.collect(data, require_grad=True)
        prediction = capture.prediction.reshape(-1)
        layer_outputs = tuple(layer.routed_output for layer in capture.layers)
        gradients_by_target: list[Sequence[Any]] = []
        for target_index in range(int(prediction.numel())):
            values = torch.autograd.grad(
                prediction[target_index],
                layer_outputs,
                retain_graph=target_index + 1 < int(prediction.numel()),
                allow_unused=True,
            )
            gradients_by_target.append(values)
        gradients: list[Any] = []
        for layer, output in enumerate(layer_outputs):
            per_target = []
            for values in gradients_by_target:
                value = values[layer]
                per_target.append(torch.zeros_like(output) if value is None else value.detach())
            gradients.append(torch.stack(per_target, dim=0))
        return capture, gradients


def reconstruction_error(layer: HeadFields) -> float:
    """Maximum absolute error in ``wV = sum_j alpha_ij m_ij``."""

    import torch

    reconstructed = torch.einsum(
        "hij,hijd->ihd",
        torch.where(layer.mask, layer.attention, torch.zeros_like(layer.attention)),
        torch.where(
            layer.mask.unsqueeze(-1),
            layer.message,
            torch.zeros_like(layer.message),
        ),
    )
    return float(
        torch.max(torch.abs(reconstructed - layer.routed_output.detach().float())).item()
    )


def attention_mass_error(layer: HeadFields) -> float:
    import torch

    mass = torch.where(
        layer.mask, layer.attention, torch.zeros_like(layer.attention)
    ).sum(dim=-1)
    valid_query = layer.mask.any(dim=-1)
    if not bool(valid_query.any()):
        return 0.0
    return float(torch.max(torch.abs(mass[valid_query] - 1.0)).item())
