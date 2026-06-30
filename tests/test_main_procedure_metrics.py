import csv
import json

from graph_specialisation_metrics.main_procedure import choose_preferred_metric_rows, extract_test_metric


def test_extract_test_metric_uses_best_validation_row_for_history_csv(tmp_path):
    path = tmp_path / "history_s41.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mae", "test_mae"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "train_loss": 0.4, "val_mae": 0.3, "test_mae": 0.31})
        writer.writerow({"epoch": 2, "train_loss": 0.2, "val_mae": 0.1, "test_mae": 0.11})
        writer.writerow({"epoch": 3, "train_loss": 0.1, "val_mae": 0.2, "test_mae": 0.21})

    row = extract_test_metric("dense_grit", path)
    assert row is not None
    assert row["source_kind"] == "history_best_val"
    assert row["val_metric"] == 0.1
    assert row["test_metric"] == 0.11


def test_training_summary_preferred_over_history(tmp_path):
    history = tmp_path / "history_s41.csv"
    with history.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_mae", "test_mae"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "val_mae": 0.1, "test_mae": 0.2})
    summary = tmp_path / "training_summary_s41.json"
    summary.write_text(json.dumps({"best_val_mae": 0.08, "best_test_mae": 0.09}), encoding="utf-8")

    rows = [
        extract_test_metric("dense_grit", history),
        extract_test_metric("dense_grit", summary),
    ]
    selected = choose_preferred_metric_rows([row for row in rows if row is not None])
    assert len(selected) == 1
    assert selected[0]["source_kind"] == "summary"
    assert selected[0]["test_metric"] == 0.09
