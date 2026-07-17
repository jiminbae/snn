#!/usr/bin/env python3
"""Export fixed trajectories (optionally in parallel) then run CPU analysis."""

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import torch

METHODS = ("final_ce", "symmetric_kl", "selective_regression_thr0.6")
SEEDS = (3, 4, 5)


def parse_gpus(value: str | None) -> list[str]:
    if value is None:
        return [os.environ.get("CUDA_VISIBLE_DEVICES", "0")]
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ValueError("parallel GPU list must look like 0,1,2,3")
    gpus = value.split(",")
    if len(gpus) != len(set(gpus)):
        raise ValueError("parallel GPU list contains duplicates")
    return gpus


def trajectory_valid(path: Path, method: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        required = (
            "prefix_logits", "targets", "predictions", "correct",
            "true_class_probability", "sample_index",
        )
        return (
            all(key in data for key in required)
            and data["method"] == method
            and int(data["seed"]) == seed
            and data["prefix_logits"].ndim == 3
            and data["prefix_logits"].shape[1:] == (8, 10)
            and data["targets"].shape[0] == data["prefix_logits"].shape[0]
        )
    except Exception:
        return False


def build_exports(results_root: Path):
    output_root = results_root / "mechanism_analysis" / "trajectories"
    return [
        (
            results_root / method / f"seed_{seed}",
            output_root / method / f"seed_{seed}.pt",
            method,
            seed,
        )
        for method in METHODS
        for seed in SEEDS
    ]


def run_exports(results_root: Path, gpus: list[str], python_bin: str) -> None:
    pending = [
        item for item in build_exports(results_root)
        if not trajectory_valid(item[1], item[2], item[3])
    ]
    work = queue.Queue()
    for item in pending:
        work.put(item)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                run_dir, output, method, seed = work.get_nowait()
            except queue.Empty:
                return
            output.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                python_bin,
                "export_confirmatory_prefix_trajectories.py",
                "--run-dir", str(run_dir),
                "--output", str(output),
                "--device", "cuda",
            ]
            result = subprocess.run(command, env=environment)
            if result.returncode != 0:
                with lock:
                    failures.append((method, seed, gpu, result.returncode))
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"Trajectory export failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/temporal_reliability_nmnist_confirmatory",
    )
    parser.add_argument("--parallel-gpus")
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
    results_root = Path(args.results_root)
    run_exports(results_root, gpus, python_bin)
    result = subprocess.run(
        [
            python_bin,
            "analyze_temporal_reliability_mechanism.py",
            "--results-root",
            str(results_root),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

