"""Human- and machine-readable progress reporting for long methodology runs."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..carriage.env import log


class ProgressJournal:
    """Append-only JSONL progress with a lightweight heartbeat.

    The journal is deliberately outside the scientific cache contract.  It describes execution,
    not an estimand, and can therefore be reused across resumed runs with different batch sizes.
    """

    def __init__(self, path: str | Path, *, heartbeat_seconds: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _cuda_memory(self) -> dict[str, float]:
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "cuda_allocated_gb": float(torch.cuda.memory_allocated() / 1.0e9),
                    "cuda_reserved_gb": float(torch.cuda.memory_reserved() / 1.0e9),
                    "cuda_peak_gb": float(torch.cuda.max_memory_allocated() / 1.0e9),
                }
        except ImportError:
            pass
        return {}

    def _gpu_device_telemetry(self) -> dict[str, float]:
        """Sample device-wide utilization without adding overhead to every event."""

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        if visible:
            command.extend(("--id", visible))
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            values = [
                float(value.strip())
                for value in output.strip().splitlines()[0].split(",")
            ]
            return {
                "gpu_utilization_percent": values[0],
                "gpu_memory_used_gb": values[1] / 1_000.0,
                "gpu_memory_total_gb": values[2] / 1_000.0,
                "gpu_power_watts": values[3],
            }
        except (
            FileNotFoundError,
            IndexError,
            OSError,
            subprocess.SubprocessError,
            ValueError,
        ):
            return {}

    def emit(self, event: str, *, message: str | None = None, **values: Any) -> None:
        device_telemetry = (
            self._gpu_device_telemetry() if event == "heartbeat" else {}
        )
        record = {
            "time_unix": time.time(),
            "elapsed_seconds": time.monotonic() - self.started,
            "event": str(event),
            **self._state,
            **values,
            **self._cuda_memory(),
            **device_telemetry,
        }
        rendered = message or " ".join(
            f"{key}={value}"
            for key, value in record.items()
            if key
            not in {
                "time_unix",
                "elapsed_seconds",
                "event",
                "cuda_allocated_gb",
                "cuda_reserved_gb",
                "cuda_peak_gb",
                "gpu_utilization_percent",
                "gpu_memory_used_gb",
                "gpu_memory_total_gb",
                "gpu_power_watts",
            }
        )
        gpu_suffix = ""
        if "gpu_utilization_percent" in record:
            gpu_suffix = (
                f" gpu={record['gpu_utilization_percent']:.0f}%"
                f" vram={record['gpu_memory_used_gb']:.1f}/"
                f"{record['gpu_memory_total_gb']:.1f}GB"
                f" power={record['gpu_power_watts']:.0f}W"
            )
        log(
            f"[progress] {event}"
            + (f" {rendered}" if rendered else "")
            + gpu_suffix
            + f" elapsed={record['elapsed_seconds']:.1f}s"
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, default=str))
                stream.write("\n")
                stream.flush()

    def update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.emit("heartbeat")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="methodology-progress",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @contextmanager
    def component(
        self,
        name: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Iterator["ProgressJournal"]:
        previous = dict(self._state)
        self.update(component=str(name), **dict(context or {}))
        self.emit("component_start")
        try:
            yield self
        except BaseException as error:
            self.emit(
                "component_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        else:
            self.emit("component_complete")
        finally:
            with self._lock:
                self._state = previous
