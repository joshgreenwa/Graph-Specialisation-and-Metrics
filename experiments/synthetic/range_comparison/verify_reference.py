"""Check the hard-coded published values against the vendored CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

from run_replication import PUBLISHED_GRID

HERE = Path(__file__).resolve().parent
FILES = {"dirac": "dirac", "rectangle": "rectangle", "power_loops": "power"}


def main() -> None:
    worst = 0.0
    for task, stem in FILES.items():
        rows = list(csv.DictReader((HERE / "reference" / f"grid_task_range_{stem}.csv").open()))
        published = {int(r["k"]): float(r["Val/TaskRangeSPDNorm"]) for r in rows}
        for index, value in enumerate(PUBLISHED_GRID[task], start=1):
            gap = abs(published[index] - value)
            worst = max(worst, gap)
            assert gap < 1e-5, f"{task} k={index}: hard-coded {value} vs CSV {published[index]}"
        print(f"{task:>12}: {len(PUBLISHED_GRID[task])} values match the vendored CSV")
    print(f"worst transcription gap: {worst:.2e}")


if __name__ == "__main__":
    main()
