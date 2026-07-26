"""Structured, flush-safe progress reporting for long Colab analyses."""

from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..carriage.env import log


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def gpu_memory_status() -> str:
    """Return a compact CUDA-memory suffix without requiring CUDA or torch."""

    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        return (
            f" | GPU {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved, "
            f"{peak:.2f} GiB peak"
        )
    except (ImportError, RuntimeError):
        return ""


def progress_kwargs(config: Any, label: str) -> dict[str, Any]:
    """Arguments shared by batched execution and bootstrap progress hooks."""

    execution = config.execution
    return {
        "progress_label": str(label),
        "progress_enabled": bool(execution.verbose_progress),
        "progress_updates": int(execution.progress_updates),
    }


@dataclass
class ProgressTracker:
    """Report bounded progress updates with elapsed time, throughput, and ETA."""

    label: str
    total: int
    unit: str = "items"
    enabled: bool = True
    updates: int = 20
    started: float = field(default_factory=time.monotonic)
    completed: int = 0
    _next_report: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total))
        self.updates = max(1, int(self.updates))
        if self.enabled:
            log(
                f"[progress] START {self.label} | {self.total} {self.unit}"
                f"{gpu_memory_status()}"
            )
        if self.total == 0:
            self._next_report = 0
        else:
            self._next_report = min(self.total, self.interval)

    @property
    def interval(self) -> int:
        return max(1, int(math.ceil(self.total / self.updates)))

    def advance(self, count: int = 1, *, detail: str | None = None) -> None:
        self.completed = min(self.total, self.completed + max(0, int(count)))
        if not self.enabled:
            return
        if self.completed < self._next_report and self.completed < self.total:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.completed / elapsed
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else math.inf
        percent = 100.0 if self.total == 0 else 100.0 * self.completed / self.total
        suffix = f" | {detail}" if detail else ""
        log(
            f"[progress] {self.label} | {self.completed}/{self.total} {self.unit} "
            f"({percent:.1f}%) | elapsed {format_duration(elapsed)} | "
            f"ETA {format_duration(eta) if math.isfinite(eta) else '--:--'} | "
            f"{rate:.2f} {self.unit}/s{suffix}{gpu_memory_status()}"
        )
        self._next_report = min(
            self.total,
            max(self.completed + 1, self._next_report + self.interval),
        )

    def finish(self, *, detail: str | None = None) -> None:
        if self.completed < self.total:
            self.advance(self.total - self.completed, detail=detail)
        elif self.enabled and self.total == 0:
            elapsed = time.monotonic() - self.started
            log(
                f"[progress] DONE {self.label} | no {self.unit} | "
                f"elapsed {format_duration(elapsed)}{gpu_memory_status()}"
            )

    def fail(self, *, detail: str | None = None) -> None:
        if not self.enabled:
            return
        elapsed = time.monotonic() - self.started
        suffix = f" | {detail}" if detail else ""
        log(
            f"[progress] FAILED {self.label} | {self.completed}/{self.total} {self.unit} | "
            f"elapsed {format_duration(elapsed)}{suffix}{gpu_memory_status()}"
        )


@contextmanager
def timed_stage(
    label: str,
    *,
    enabled: bool = True,
    heartbeat_seconds: float = 60.0,
) -> Iterator[None]:
    """Log stage boundaries and a heartbeat while one blocking operation is running."""

    started = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        interval = max(1.0, float(heartbeat_seconds))
        while not stop.wait(interval):
            elapsed = time.monotonic() - started
            log(
                f"[progress] STILL RUNNING {label} | elapsed "
                f"{format_duration(elapsed)}{gpu_memory_status()}"
            )

    thread = None
    if enabled:
        log(f"[progress] START {label}{gpu_memory_status()}")
        if float(heartbeat_seconds) > 0:
            thread = threading.Thread(
                target=heartbeat,
                name=f"progress:{label}",
                daemon=True,
            )
            thread.start()
    try:
        yield
    except BaseException:
        if enabled:
            elapsed = time.monotonic() - started
            log(
                f"[progress] FAILED {label} | elapsed {format_duration(elapsed)}"
                f"{gpu_memory_status()}"
            )
        raise
    else:
        if enabled:
            elapsed = time.monotonic() - started
            log(
                f"[progress] DONE {label} | elapsed {format_duration(elapsed)}"
                f"{gpu_memory_status()}"
            )
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=0.2)
