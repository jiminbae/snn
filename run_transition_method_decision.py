#!/usr/bin/env python3
"""Export hidden split trajectories in parallel, then run the branch decision."""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import torch

from export_split_trajectories import sha256_file


SEEDS = (3, 4, 5)
REQUIRED_RUN_FILES = (
    "config.json",
    "best_checkpoint.pt",
    "split_indices.pt",
    "selection_summary.json",
    "summary.json",
)


def parse_gpus(value: str | None) -> list[str]:
    resolved = value or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2"
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", resolved):
        raise ValueError("parallel GPU list must look like 0,1,2")
    gpus = resolved.split(",")
    if len(gpus) != len(set(gpus)):
        raise ValueError("parallel GPU list contains duplicates")
    return gpus


def current_fingerprints(run_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(run_dir / name)
        for name in REQUIRED_RUN_FILES
    }


def split_export_valid(run_dir: Path) -> bool:
    if any(not (run_dir / name).is_file() for name in REQUIRED_RUN_FILES):
        return False
    expected = current_fingerprints(run_dir)
    trajectory_root = run_dir / "trajectories"
    try:
        for split in ("train", "val", "test"):
            path = trajectory_root / f"{split}_trajectories.pt"
            if not path.is_file():
                return False
            payload: dict[str, Any] = torch.load(
                path, map_location="cpu", weights_only=False
            )
            hidden = payload.get("hidden_features")
            hidden_metadata = payload.get("hidden_feature_metadata", {})
            metadata = payload.get("metadata", {})
            if (
                payload.get("split") != split
                or not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 3
                or hidden.shape[:2] != payload["prefix_logits"].shape[:2]
                or not torch.isfinite(hidden).all()
                or hidden_metadata.get("format_version") != 1
                or hidden_metadata.get("uses_target") is not False
                or hidden_metadata.get("causal") is not True
                or metadata.get("source_fingerprints") != expected
                or metadata.get("hidden_features_included") is not True
            ):
                return False
        return True
    except Exception:
        return False


def run_exports(
    results_root: Path,
    gpus: list[str],
    python_bin: str,
    *,
    batch_size: int,
    num_workers: int,
) -> None:
    pending = [
        results_root / "final_ce" / f"seed_{seed}"
        for seed in SEEDS
        if not split_export_valid(results_root / "final_ce" / f"seed_{seed}")
    ]
    work: queue.Queue[Path] = queue.Queue()
    for run_dir in pending:
        work.put(run_dir)
    failures: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                run_dir = work.get_nowait()
            except queue.Empty:
                return
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                python_bin,
                "export_split_trajectories.py",
                "--run-dir",
                str(run_dir),
                "--device",
                "cuda",
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(num_workers),
                "--include-hidden-features",
            ]
            result = subprocess.run(command, env=environment)
            if result.returncode != 0:
                with lock:
                    failures.append((run_dir.name, gpu, result.returncode))
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Hidden split trajectory export failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/temporal_reliability_nmnist_confirmatory"),
    )
    parser.add_argument("--parallel-gpus")
    parser.add_argument("--export-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--analysis-device", default="cuda")
    parser.add_argument("--direction-epochs", type=int, default=200)
    parser.add_argument("--hidden-epochs", type=int, default=40)
    parser.add_argument("--hidden-batch-size", type=int, default=1024)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    try:
        gpus = parse_gpus(args.parallel_gpus)
    except ValueError as error:
        parser.error(str(error))

    python_bin = (
        ".venv/bin/python"
        if Path(".venv/bin/python").is_file()
        else sys.executable
    )
    if not args.skip_export:
        run_exports(
            args.results_root,
            gpus,
            python_bin,
            batch_size=args.export_batch_size,
            num_workers=args.num_workers,
        )

    command = [
        python_bin,
        "analyze_transition_method_decision.py",
        "--results-root",
        str(args.results_root),
        "--device",
        args.analysis_device,
        "--direction-epochs",
        str(args.direction_epochs),
        "--hidden-epochs",
        str(args.hidden_epochs),
        "--hidden-batch-size",
        str(args.hidden_batch_size),
    ]
    environment = os.environ.copy()
    if args.analysis_device.startswith("cuda"):
        environment["CUDA_VISIBLE_DEVICES"] = gpus[0]
    result = subprocess.run(command, env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
