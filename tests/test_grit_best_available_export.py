import importlib.util
import os
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments/grit_hpc/bin/export_best_available.py"
SPEC = importlib.util.spec_from_file_location("grit_best_available", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_log(path: Path, values: list[tuple[int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for epoch, val, test in values:
        lines.extend(
            (
                f"train: {{'epoch': {epoch}, 'mae': {val + 1.0}}}",
                f"val: {{'epoch': {epoch}, 'mae': {val}}}",
                f"test: {{'epoch': {epoch}, 'mae': {test}}}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ckpt(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_selects_exact_stable_best_checkpoint(tmp_path):
    run = tmp_path / "zinc.dense.s0"
    write_log(run / "wrapper_logs/run.log", [(10, 0.2, 0.3), (11, 0.1, 0.2), (20, 0.15, 0.25)])
    recovery = run / "results/_recovery_checkpoints/seed0"
    write_ckpt(recovery / "best.ckpt", b"exact-best")
    (recovery / "best_epoch.txt").write_text("11\n", encoding="utf-8")
    write_ckpt(recovery / "period_10_epoch10.ckpt", b"ten")
    write_ckpt(recovery / "period_10_epoch20.ckpt", b"twenty")

    selected = MODULE.select_run(run)

    assert selected["selected_epoch"] == 11
    assert selected["exact_global_best"] is True
    assert selected["source_checkpoint"].endswith("best.ckpt")


def test_selects_lowest_validation_mae_among_saved_epochs(tmp_path):
    run = tmp_path / "qm9_gap.dense.s0"
    write_log(run / "wrapper_logs/old.log", [(280, 0.3, 0.4), (290, 0.2, 0.25)])
    newer = run / "wrapper_logs/new.log"
    write_log(newer, [(293, 0.1, 0.15), (299, 0.22, 0.24)])
    old_time = (run / "wrapper_logs/old.log").stat().st_mtime
    os.utime(newer, (old_time + 1, old_time + 1))
    recovery = run / "results/_recovery_checkpoints/seed0"
    write_ckpt(recovery / "period_10_epoch280.ckpt", b"280")
    write_ckpt(recovery / "period_10_epoch290.ckpt", b"290")
    write_ckpt(recovery / "latest.ckpt", b"299")
    (recovery / "latest_epoch.txt").write_text("299\n", encoding="utf-8")

    selected = MODULE.select_run(run)

    assert selected["global_log_best_epoch"] == 293
    assert selected["selected_epoch"] == 290
    assert selected["exact_global_best"] is False
    assert selected["selected_val_mae"] == 0.2


def test_exports_checkpoint_and_provenance(tmp_path):
    input_root = tmp_path / "input"
    run = input_root / "zinc.1hop.s2"
    write_log(run / "wrapper_logs/run.log", [(30, 0.2, 0.3)])
    source = run / "results/_recovery_checkpoints/seed2/period_10_epoch30.ckpt"
    write_ckpt(source, b"checkpoint-data")
    output = tmp_path / "export"

    records = MODULE.export_runs(input_root, output, [run.name])

    copied = output / "checkpoints/zinc.1hop.s2/best_available.ckpt"
    assert copied.read_bytes() == b"checkpoint-data"
    assert (output / "manifest.json").is_file()
    assert (output / "manifest.tsv").is_file()
    assert records[0]["source_checkpoint_sha256"] == MODULE.sha256(copied)
