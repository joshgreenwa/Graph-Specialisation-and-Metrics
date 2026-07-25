"""Thin canonical adapter around the checkpoint-compatible GRIT machinery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .audit import audit_check


@dataclass
class BackendCapture:
    prediction: Any
    z: Any
    target: Any
    transport: tuple[Any, ...]
    final_state: Any
    real_mask: Any


@dataclass
class CleanJacobians:
    capture: BackendCapture
    transport: Any
    final_state: Any


class CanonicalGritBackend:
    """Expose only the native sites required by the final methodology."""

    def __init__(self, grit_model: Any, task: Any, sigma: Sequence[float]):
        import numpy as np

        self.gm = grit_model
        self.task = task
        self.sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        self.has_virtual_node = False

    @property
    def geometry(self) -> dict[str, int]:
        return {
            "layers": int(self.gm.L),
            "heads": int(self.gm.H),
            "head_width": int(self.gm.dh),
            "hidden_width": int(self.gm.dim_h),
            "outputs": int(len(self.sigma)),
        }

    @property
    def special_carrier_labels(self) -> tuple[str, ...]:
        return ("virtual",) if self.task.virtual_node else ()

    def _z(self, prediction):
        return self.task.output.transform(prediction, self.sigma)

    def capture(
        self,
        data_list: Sequence[Any],
        *,
        require_grad: bool,
        include_virtual_transport: bool = True,
    ) -> BackendCapture:
        from torch_geometric.data import Batch

        if not data_list:
            raise ValueError("capture requires at least one graph")
        if require_grad and len(data_list) != 1:
            raise ValueError("clean Jacobians are captured one graph at a time")
        counts = [int(data.num_nodes) for data in data_list]
        if len(set(counts)) != 1:
            raise ValueError("one event capture must contain replicas of the same base graph")
        batch = Batch.from_data_list([data.clone() for data in data_list]).to(self.gm.device)
        final: dict[str, Any] = {}

        def final_hook(_module, _inputs, output):
            full = output.x
            mask = getattr(output, "real_node_mask", None)
            final["full"] = full
            final["real_mask"] = mask.detach().clone() if mask is not None else None
            # Store the pre-pooling real-node state now; post_mp may mutate the Batch later.
            final["real"] = full[mask] if mask is not None else full

        handle = self.gm.model.model.layers.register_forward_hook(final_hook)
        try:
            captured = self.gm.capture(
                batch,
                want_grad=require_grad,
                want_attn=False,
                include_virtual_transport=include_virtual_transport,
            )
        finally:
            handle.remove()
        if "full" not in final:
            raise RuntimeError("final-state hook did not fire")
        prediction = captured["pred"]
        z = self._z(prediction)
        repetitions = len(data_list)
        real_nodes = counts[0]
        real_mask = final["real_mask"]
        self.has_virtual_node = self.has_virtual_node or real_mask is not None
        if bool(real_mask is not None) != bool(self.task.virtual_node):
            raise RuntimeError(
                f"task registration virtual_node={self.task.virtual_node} disagrees with "
                f"the loaded GRIT model (real_node_mask present={real_mask is not None})"
            )

        transport: list[Any] = []
        for layer in captured["wV"]:
            if require_grad:
                transport.append(layer)
            else:
                carriers = int(layer.shape[0]) // repetitions
                transport.append(layer.reshape(repetitions, carriers, self.gm.H, self.gm.dh))
        if require_grad:
            final_state = final["full"]
        else:
            final_state = final["real"].reshape(repetitions, real_nodes, self.gm.dim_h)
        return BackendCapture(
            prediction=prediction,
            z=z,
            target=captured["true"],
            transport=tuple(transport),
            final_state=final_state,
            real_mask=real_mask,
        )

    def clean_jacobians(self, data: Any) -> CleanJacobians:
        import torch

        capture = self.capture([data], require_grad=True)
        targets = tuple(capture.transport) + (capture.final_state,)
        z = capture.z.reshape(-1)
        by_output: list[tuple[Any, ...]] = []
        for output in range(int(z.numel())):
            gradients = torch.autograd.grad(
                z[output],
                targets,
                retain_graph=output + 1 < int(z.numel()),
                allow_unused=False,
            )
            by_output.append(gradients)
        transport = torch.stack(
            [
                torch.stack([row[layer].detach() for row in by_output], dim=0)
                for layer in range(int(self.gm.L))
            ],
            dim=1,
        )  # [T,L,N,H,D]
        final_gradient = torch.stack(
            [row[-1].detach() for row in by_output], dim=0
        )  # [T,N(+v),M]
        final_state = capture.final_state
        if capture.real_mask is not None:
            final_gradient = final_gradient[:, capture.real_mask]
            final_state = final_state[capture.real_mask]
        audit_check(
            bool(torch.isfinite(transport).all() and torch.isfinite(final_gradient).all()),
            "backend.finite_clean_jacobian",
            "non-finite clean z-space Jacobian; downstream scores inherit the non-finite entries",
        )
        layer_norms = torch.linalg.vector_norm(
            transport.reshape(transport.shape[0], transport.shape[1], -1), dim=(0, 2)
        )
        if bool((layer_norms <= 0).any()):
            missing = torch.nonzero(layer_norms <= 0).reshape(-1).tolist()
            audit_check(
                False,
                "backend.nonzero_clean_transport",
                f"zero clean transport Jacobian in layers {missing}; those layers score zero",
                context={"layers": missing},
            )
        capture.final_state = final_state
        return CleanJacobians(capture, transport, final_gradient)

    def loss_per_graph(self, prediction, target):
        return self.task.loss_per_graph(
            prediction.reshape(prediction.shape[0], -1),
            target.reshape(target.shape[0], -1),
        )

    def loss_from_pooled(self, target):
        """Differentiable exact pooling-to-readout replay for path integration."""

        from ..carriage.grit_runner import _pooled_head_predictions

        def evaluate(pooled):
            prediction = _pooled_head_predictions(self.gm.model, pooled, target)
            return self.loss_per_graph(prediction, target.expand(prediction.shape[0], -1))

        return evaluate

    def carriage_weights(self, data: Any, final_state) -> Any:
        """Linear real-node pooling weights used by the exact carriage integral."""

        import torch

        pooling = str(self.gm.cfg.model.graph_pooling)
        count = int(final_state.shape[-2])
        weights = torch.ones(count, device=final_state.device, dtype=final_state.dtype)
        if pooling == "mean":
            weights /= float(count)
        elif pooling != "add":
            raise ValueError(f"canonical carriage requires add/mean pooling, got {pooling!r}")
        return weights

    def transport_distances(self, data: Any, source: int, pristine) -> list[Any]:
        distances: list[Any] = list(pristine[int(source), :])
        if self.task.virtual_node:
            distances.append("virtual")
        return distances

    def carriage_distance_matrix(self, data: Any, sources: Sequence[int], pristine):
        import numpy as np

        return pristine[np.asarray(sources, dtype=np.int64), :].T

    def carriage_carrier_kind(self, data: Any, carrier: int) -> str:
        del data, carrier
        return "molecular_node"

    def attention_normalization_error(self, data: Any) -> float:
        import torch
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([data.clone()]).to(self.gm.device)
        captured = self.gm.capture(
            batch,
            want_grad=False,
            want_attn=True,
            include_virtual_transport=True,
        )
        edge_index = captured["edge_index"].long()
        receiver = edge_index[1]
        error = 0.0
        for attention in captured["attn"]:
            mass = torch.zeros(
                int(captured["wV"][0].shape[0]),
                int(self.gm.H),
                dtype=attention.dtype,
                device=attention.device,
            )
            mass.index_add_(0, receiver, attention)
            valid = torch.unique(receiver)
            error = max(error, float(torch.max(torch.abs(mass[valid] - 1.0)).item()))
        return error

    def clean_attention_distance(self, data: Any, pristine, axis: Any):
        """Normalize clean sparse attention by head and graph."""

        import numpy as np
        from torch_geometric.data import Batch

        batch = Batch.from_data_list([data.clone()]).to(self.gm.device)
        captured = self.gm.capture(
            batch,
            want_grad=False,
            want_attn=True,
            include_virtual_transport=True,
        )
        edge_index = captured["edge_index"].detach().cpu().numpy()
        buckets = []
        for sender, receiver in edge_index.T:
            if int(sender) >= int(data.num_nodes) or int(receiver) >= int(data.num_nodes):
                buckets.append(axis.index("virtual"))
            else:
                buckets.append(axis.index(pristine[int(receiver), int(sender)]))
        buckets = np.asarray(buckets, dtype=np.int64)
        profile = np.zeros(
            (int(self.gm.L), int(self.gm.H), len(axis.labels)),
            dtype=np.float64,
        )
        for layer, attention in enumerate(captured["attn"]):
            values = attention.detach().cpu().numpy()
            for distance_index in range(len(axis.labels)):
                mask = buckets == distance_index
                if mask.any():
                    profile[layer, :, distance_index] = values[mask].sum(axis=0)
            denominator = profile[layer].sum(axis=-1, keepdims=True)
            np.divide(
                profile[layer],
                denominator,
                out=profile[layer],
                where=denominator > 0,
            )
        return profile

    def _native_forward(
        self,
        data_list: Sequence[Any],
        *,
        family: Sequence[tuple[int, int]] = (),
        replacements: Sequence[Any] | None = None,
        ablate: bool = False,
    ):
        """Forward with zeroing or replacement at the actual attention output site."""

        import torch
        from torch_geometric.data import Batch

        family = tuple((int(layer), int(head)) for layer, head in family)
        by_layer: dict[int, list[int]] = {}
        for layer, head in family:
            by_layer.setdefault(layer, []).append(head)
        if replacements is not None and len(replacements) != int(self.gm.L):
            raise ValueError("replacement activations must contain one tensor per layer")
        handles = []
        for layer, heads in by_layer.items():
            heads = sorted(set(heads))

            def make_hook(layer_index, selected_heads):
                def hook(_module, _inputs, output):
                    routed, edge = output
                    changed = routed.clone()
                    if ablate:
                        changed[:, selected_heads, :] = 0.0
                    else:
                        donor = replacements[layer_index].to(
                            device=changed.device, dtype=changed.dtype
                        )
                        if donor.shape != changed.shape:
                            raise RuntimeError(
                                f"patch geometry differs at layer {layer_index}: "
                                f"{tuple(donor.shape)} vs {tuple(changed.shape)}"
                            )
                        changed[:, selected_heads, :] = donor[:, selected_heads, :]
                    return changed, edge

                return hook

            handles.append(
                self.gm.attn_layers[layer].register_forward_hook(
                    make_hook(layer, heads)
                )
            )
        try:
            batch = Batch.from_data_list([data.clone() for data in data_list]).to(self.gm.device)
            with torch.no_grad():
                prediction, target = self.gm.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        return prediction, self._z(prediction), target

    def ablate(self, data_list: Sequence[Any], family: Sequence[tuple[int, int]]):
        return self._native_forward(data_list, family=family, ablate=True)

    def patch(
        self,
        target: Any,
        donor_transport: Sequence[Any],
        family: Sequence[tuple[int, int]],
    ):
        return self._native_forward(
            [target],
            family=family,
            replacements=donor_transport,
            ablate=False,
        )

    def patch_many(
        self,
        targets: Sequence[Any],
        donor_transport: Sequence[Any],
        family: Sequence[tuple[int, int]],
    ):
        return self._native_forward(
            targets,
            family=family,
            replacements=donor_transport,
            ablate=False,
        )

    def replacement_batch(
        self,
        capture: BackendCapture,
        indices: Sequence[int],
        *,
        repeat_single: bool = False,
    ) -> tuple[Any, ...]:
        """Convert graph-major captured replicas back to GRIT's native batched row order."""

        import torch

        selected = [int(value) for value in indices]
        result = []
        for layer in capture.transport:
            rows = layer[selected]
            if repeat_single and rows.shape[0] == 1 and len(selected) > 1:
                rows = rows.expand(len(selected), *rows.shape[1:])
            if self.has_virtual_node:
                real = rows[:, :-1].reshape(-1, rows.shape[-2], rows.shape[-1])
                virtual = rows[:, -1]
                native = torch.cat((real, virtual), dim=0)
            else:
                native = rows.reshape(-1, rows.shape[-2], rows.shape[-1])
            result.append(native.detach())
        return tuple(result)
