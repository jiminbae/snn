#!/usr/bin/env python3
"""Run one shared anytime SNN and matched fixed-prefix specialists."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ANCHOR_TIMESTEPS = (1, 2, 4, 6, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prefix diagnostics and specialist comparisons.")
    parser.add_argument("--dataset", choices=["nmnist", "dvs_gesture"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--event-frame-mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--event-downsample-size", type=int, default=None)
    parser.add_argument("--results-dir", default="results/prefix_diagnostics")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run(args: argparse.Namespace, run_name: str, prefix_steps: int, diagnostics: bool) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    summary_path = results_dir / run_name / "summary.json"
    run_dir = summary_path.parent
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Run already exists: {run_dir}. Use --overwrite or choose a new results directory."
            )
        shutil.rmtree(run_dir)
    command = [
            sys.executable,
            str(root / "train.py"),
            "--model", "fixed_lif",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--tmax", str(args.tmax),
            "--device", args.device,
            "--seed", str(args.seed),
            "--event-frame-mode", args.event_frame_mode,
            "--data-dir", args.data_dir,
            "--results-dir", str(results_dir),
            "--run-name", run_name,
    ]
    if args.event_downsample_size is not None:
        command.extend(["--event-downsample-size", str(args.event_downsample_size)])
    if prefix_steps < args.tmax:
        command.extend(["--temporal-prefix-mode", "truncate", "--temporal-prefix-steps", str(prefix_steps)])
    if diagnostics:
        command.append("--prefix-diagnostics")
    if args.amp:
        command.append("--amp")
    for name in ("num_workers", "limit_train_batches", "limit_test_batches"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    subprocess.run(command, cwd=root, check=True)
    return _load_json(summary_path)


def main() -> None:
    args = parse_args()
    if args.tmax < max(ANCHOR_TIMESTEPS):
        raise ValueError(f"tmax must be at least {max(ANCHOR_TIMESTEPS)} for the default anchors.")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    shared = _run(args, f"shared_fixed_lif_T{args.tmax}", args.tmax, diagnostics=True)
    rows: list[dict[str, Any]] = []
    specialist_accuracies: dict[str, float] = {}
    for timestep in ANCHOR_TIMESTEPS:
        shared_accuracy = float(shared[f"prefix_accuracy_t{timestep}"])
        if timestep == args.tmax:
            specialist_accuracy = shared_accuracy
            reference = "shared_same_budget"
        else:
            specialist = _run(args, f"specialist_T{timestep}", timestep, diagnostics=False)
            specialist_accuracy = float(specialist["test_accuracy"])
            reference = "independent_specialist"
        regret = specialist_accuracy - shared_accuracy
        specialist_accuracies[f"specialist_accuracy_t{timestep}"] = specialist_accuracy
        rows.append({
            "Timestep": timestep,
            "Shared Accuracy": shared_accuracy,
            "Specialist Accuracy": specialist_accuracy,
            "Prefix Regret": regret,
            "Reference": reference,
        })

    with (results_dir / "prefix_regret.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    regrets = [row["Prefix Regret"] for row in rows if row["Timestep"] != args.tmax]
    diagnostic_summary = {
        "shared_run": shared["run_name"],
        "mean_prefix_regret": sum(regrets) / len(regrets),
        "max_prefix_regret": max(regrets),
        **specialist_accuracies,
    }
    with (results_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, indent=2, sort_keys=True)
    print(f"Saved prefix diagnostics to: {results_dir}")


if __name__ == "__main__":
    main()
