"""CSV and JSON logging utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def prepare_run_dir(results_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(results_dir) / run_name
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_last_metrics(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}
