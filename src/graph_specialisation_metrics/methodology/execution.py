"""Runtime-only batching helpers that do not alter scientific estimands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class BatchExecutionReport:
    requested_graphs_per_batch: int
    successful_batches: int
    minimum_graphs_per_batch: int
    maximum_graphs_per_batch: int
    oom_retries: int


def is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, RuntimeError) and (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "cuda out of memory" in text
    )


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def execute_graph_batches(
    items: Sequence[Any],
    *,
    graphs_per_batch: int,
    execute: Callable[[Sequence[Any]], Any],
    consume: Callable[[Any], None],
    oom_backoff: bool,
    item_cost: Callable[[Any], int] | None = None,
    max_cost: int | None = None,
    on_batch: Callable[[int, int, int], None] | None = None,
) -> BatchExecutionReport:
    """Execute ordered graph groups, halving only the failing batch on CUDA OOM.

    ``execute`` must not mutate result accumulators. ``consume`` is called only after the complete
    batch succeeds, so an OOM retry cannot duplicate partial scientific observations.
    """

    requested = int(graphs_per_batch)
    if requested < 1:
        raise ValueError("graphs_per_batch must be positive")
    values = list(items)
    cursor = 0
    successful_sizes: list[int] = []
    retries = 0
    while cursor < len(values):
        size = min(requested, len(values) - cursor)
        if item_cost is not None and max_cost is not None:
            while size > 1 and sum(
                max(1, int(item_cost(value)))
                for value in values[cursor : cursor + size]
            ) > int(max_cost):
                size -= 1
        while True:
            chunk = values[cursor : cursor + size]
            try:
                result = execute(chunk)
            except RuntimeError as error:
                if not oom_backoff or size <= 1 or not is_cuda_oom(error):
                    raise
                retries += 1
                size = max(1, size // 2)
                del error
                _clear_cuda_cache()
                continue
            consume(result)
            successful_sizes.append(size)
            cursor += size
            if on_batch is not None:
                on_batch(cursor, len(values), size)
            break
    return BatchExecutionReport(
        requested_graphs_per_batch=requested,
        successful_batches=len(successful_sizes),
        minimum_graphs_per_batch=min(successful_sizes, default=0),
        maximum_graphs_per_batch=max(successful_sizes, default=0),
        oom_retries=retries,
    )
