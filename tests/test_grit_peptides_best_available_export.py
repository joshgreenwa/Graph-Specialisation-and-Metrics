import importlib.util
import os
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments/grit_hpc/bin/export_peptides_best_available.py"
SPEC = importlib.util.spec_from_file_location("grit_peptides_best_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_log(path: Path, metric: str, values: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for epoch, validation, test in values:
        lines.extend(
            (
                f"train: {{'epoch': {epoch}, '{metric}': {validation}}}",
                f"val: {{'epoch': {epoch}, '{metric}': {validation}}}",
                f"test: {{'epoch': {epoch}, '{metric}': {test}}}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ckpt(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_func_maximizes_validation_ap_over_saved_epochs(tmp_path):
    run = tmp_path / "peptides_func.dense.s0"
    write_log(
        run / "wrapper_logs/run.log",
        "ap",
        [(10, 0.4, 0.3), (11, 0.8, 0.7), (20, 0.6, 0.5), (199, 0.2, 0.2)],
    )
    recovery = run / "results/_recovery_checkpoints/seed0"
    write_ckpt(recovery / "period_10_epoch10.ckpt", b"ten")
    write_ckpt(recovery / "period_10_epoch20.ckpt", b"twenty")

    selected = MODULE.select_run(run)

    assert selected["global_log_best_epoch"] == 11
    assert selected["selected_epoch"] == 20
    assert selected["selected_validation"] == 0.6
    assert selected["exact_global_best"] is False


def test_struct_minimizes_validation_mae_and_uses_stable_best(tmp_path):
    run = tmp_path / "peptides_struct.1hop.s2"
    old = run / "wrapper_logs/old.log"
    new = run / "wrapper_logs/new.log"
    write_log(old, "mae", [(10, 0.4, 0.5), (11, 0.2, 0.3)])
    write_log(new, "mae", [(20, 0.3, 0.4), (199, 0.35, 0.45)])
    old_time = old.stat().st_mtime
    os.utime(new, (old_time + 1, old_time + 1))
    recovery = run / "results/_recovery_checkpoints/seed2"
    write_ckpt(recovery / "best.ckpt", b"best")
    (recovery / "best_epoch.txt").write_text("11\n", encoding="utf-8")
    write_ckpt(recovery / "period_10_epoch20.ckpt", b"twenty")

    selected = MODULE.select_run(run)

    assert selected["global_log_best_epoch"] == 11
    assert selected["selected_epoch"] == 11
    assert selected["selected_test"] == 0.3
    assert selected["exact_global_best"] is True


def test_summary_reports_task_specific_metrics():
    records = []
    for task, metric, direction in (
        ("peptides_func", "ap", "max"),
        ("peptides_struct", "mae", "min"),
    ):
        for variant in MODULE.VARIANTS:
            for seed in range(3):
                records.append(
                    {
                        "task": task,
                        "variant": variant,
                        "seed": seed,
                        "selection_metric": metric,
                        "selection_direction": direction,
                        "global_log_best_epoch": 10,
                        "selected_epoch": 10,
                        "selected_validation": 0.5 + seed * 0.01,
                        "selected_test": 0.4 + seed * 0.01,
                        "exact_global_best": True,
                    }
                )

    summary = MODULE.format_summary(records)

    assert "Validation AP ↑" in summary
    assert "Test MAE ↓" in summary
    assert summary.count("3/3") == 10
