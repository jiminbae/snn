#!/usr/bin/env python3
"""Launch independent N-MNIST confirmatory runs on one or more GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

METHODS = ("final_ce", "symmetric_kl", "selective_regression")
SEEDS = (3, 4, 5)
COMPLETION_FILES = (
    "temporal_reliability_summary.json",
    "config.json",
    "best_checkpoint.pt",
    "prefix_metrics.json",
)


@dataclass(frozen=True)
class Job:
    method: str
    label: str
    seed: int
    run_dir: Path

    def command(self, python_bin: str, device: str = "cuda") -> list[str]:
        command = [
            python_bin,
            "train.py",
            "--dataset", "nmnist",
            "--model", "fixed_lif",
            "--epochs", "30",
            "--batch-size", "32",
            "--tmax", "8",
            "--event-frame-mode", "binary",
            "--val-ratio", "0.2",
            "--split-seed", "123",
            "--checkpoint-selection", "best_val",
            "--selection-metric", "val_acc",
            "--prefix-diagnostics",
            "--prefix-loss-weight", "0.0",
            "--temporal-loss-weight", "1.0",
            "--temporal-margin", "0.0",
            "--temporal-temperature", "1.0",
            "--temporal-selection-mode", "hard",
            "--device", device,
            "--data-dir", "data",
            "--seed", str(self.seed),
            "--temporal-training-mode", self.method,
            "--results-dir", str(self.run_dir.parent),
            "--run-name", self.run_dir.name,
        ]
        if self.method == "selective_regression":
            command.extend(["--temporal-confidence-threshold", "0.6"])
        return command


def parse_gpu_list(value: str) -> list[str]:
    if not value or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ValueError(
            "GPU list must be comma-separated non-negative integers, for example 0,1,2,3"
        )
    gpus = value.split(",")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list contains duplicates: {value}")
    return gpus


def build_jobs(results_dir: Path) -> list[Job]:
    jobs = []
    for method in METHODS:
        label = (
            "selective_regression_thr0.6"
            if method == "selective_regression"
            else method
        )
        for seed in SEEDS:
            jobs.append(Job(method, label, seed, results_dir / label / f"seed_{seed}"))
    return jobs


def is_complete(job: Job) -> bool:
    return all((job.run_dir / filename).is_file() for filename in COMPLETION_FILES)


def pending_jobs(jobs: list[Job]) -> tuple[list[Job], list[Job]]:
    complete = [job for job in jobs if is_complete(job)]
    pending = [job for job in jobs if job not in complete]
    return pending, complete


def dry_run_plan(jobs: list[Job], gpus: list[str]) -> list[tuple[Job, str]]:
    return [(job, gpus[index % len(gpus)]) for index, job in enumerate(jobs)]


def validate_gpu_indices(gpus: list[str]) -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    count = sum(line.startswith("GPU ") for line in result.stdout.splitlines())
    invalid = [gpu for gpu in gpus if int(gpu) >= count]
    if invalid:
        raise ValueError(
            f"GPU indices {invalid} do not exist; nvidia-smi reports {count} GPU(s)"
        )


Runner = Callable[[Job, str], bool]


def run_job_pool(
    jobs: list[Job],
    gpus: list[str],
    runner: Runner,
) -> list[tuple[Job, str, bool]]:
    work: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work.put(job)
    stop = threading.Event()
    results: list[tuple[Job, str, bool]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while not stop.is_set():
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            success = runner(job, gpu)
            with lock:
                results.append((job, gpu, success))
            work.task_done()
            if not success:
                stop.set()
                return

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


class Launcher:
    def __init__(self, results_dir: Path, python_bin: str) -> None:
        self.results_dir = results_dir
        self.python_bin = python_bin
        self.active: dict[int, subprocess.Popen[str]] = {}
        self.active_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.launcher_log = results_dir / "launcher.log"

    def log(self, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        with self.log_lock:
            print(line, flush=True)
            with self.launcher_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def terminate_children(self) -> None:
        with self.active_lock:
            processes = list(self.active.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def run_job(self, job: Job, gpu: str) -> bool:
        command = job.command(self.python_bin)
        return self.run_command(job, gpu, command)

    def run_command(
        self,
        job: Job,
        gpu: str,
        command: list[str],
    ) -> bool:
        job.run_dir.mkdir(parents=True, exist_ok=True)
        command_text = shlex.join(command)
        self.log(f"START method={job.label} seed={job.seed} gpu={gpu}")
        with (job.run_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"$ CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} {command_text}\n")
            handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
                start_new_session=True,
            )
            with self.active_lock:
                self.active[process.pid] = process
            return_code = process.wait()
            with self.active_lock:
                self.active.pop(process.pid, None)
        success = return_code == 0
        status = "SUCCESS" if success else f"FAILED exit={return_code}"
        self.log(
            f"{status} method={job.label} seed={job.seed} gpu={gpu}"
        )
        if not success:
            self.terminate_children()
        return success


def prepare_nmnist_data(data_dir: str = "data") -> None:
    import tonic

    train = tonic.datasets.NMNIST(save_to=data_dir, train=True)
    test = tonic.datasets.NMNIST(save_to=data_dir, train=False)
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("N-MNIST preparation produced an empty dataset")


def smoke_command(python_bin: str, results_dir: Path, device: str) -> tuple[Job, list[str]]:
    job = Job("final_ce", "final_ce", 3, results_dir / "final_ce" / "seed_3_smoke")
    command = job.command(python_bin, device=device)
    epoch_index = command.index("--epochs") + 1
    command[epoch_index] = "1"
    command.extend(
        [
            "--limit-train-batches", "2",
            "--limit-val-batches", "2",
            "--limit-test-batches", "2",
            "--num-workers", "0",
        ]
    )
    return job, command


def aggregate_command(python_bin: str, jobs: list[Job], results_dir: Path) -> list[str]:
    return [
        python_bin,
        "aggregate_temporal_reliability_confirmatory.py",
        "--run-dirs",
        *[str(job.run_dir) for job in jobs],
        "--output-dir",
        str(results_dir / "aggregate"),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed N-MNIST confirmatory experiment.",
        epilog=(
            "Explicit --parallel-gpus indices refer to physical nvidia-smi indices. "
            "Without it, CONFIRM_GPUS is used; otherwise serial mode preserves the "
            "caller's CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results/temporal_reliability_nmnist_confirmatory",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--parallel-gpus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and args.parallel_gpus is not None:
        parser.error("--smoke and --parallel-gpus cannot be used together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    python_bin = os.environ.get("PYTHON_BIN", ".venv/bin/python")
    if not Path(python_bin).is_file():
        python_bin = sys.executable

    requested = args.parallel_gpus or os.environ.get("CONFIRM_GPUS")
    if requested is not None:
        try:
            gpus = parse_gpu_list(requested)
            validate_gpu_indices(gpus)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpus = [visible] if visible and "," not in visible else ["0"]

    if args.smoke:
        gpu = gpus[0]
        if args.dry_run:
            job, command = smoke_command(python_bin, results_dir, "cuda")
            print(f"SMOKE gpu={gpu}: CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
            return 0
        results_dir.mkdir(parents=True, exist_ok=True)
        launcher = Launcher(results_dir, python_bin)
        try:
            prepare_nmnist_data()
            job, command = smoke_command(python_bin, results_dir, "cuda")
            if is_complete(job):
                launcher.log("SKIP completed smoke")
                return 0
            return 0 if launcher.run_command(job, gpu, command) else 1
        except KeyboardInterrupt:
            launcher.terminate_children()
            return 130

    jobs = build_jobs(results_dir)
    pending, complete = pending_jobs(jobs)
    if args.dry_run:
        for job in complete:
            print(f"SKIP method={job.label} seed={job.seed}")
        for job, gpu in dry_run_plan(pending, gpus):
            print(
                f"PLAN method={job.label} seed={job.seed} gpu={gpu}: "
                f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(job.command(python_bin))}"
            )
        print("DRY RUN: dataset preparation, training, and aggregate were not executed")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    launcher = Launcher(results_dir, python_bin)
    launcher.log(f"START GPUs={','.join(gpus)} jobs={len(jobs)} pending={len(pending)}")
    for job in jobs:
        launcher.log(f"JOB method={job.label} seed={job.seed}")
    for job in complete:
        launcher.log(f"SKIP completed method={job.label} seed={job.seed}")
    try:
        prepare_nmnist_data()
        launcher.log("DATASET preparation successful")
        outcomes = run_job_pool(pending, gpus, launcher.run_job)
        failures = [item for item in outcomes if not item[2]]
        if failures or len(outcomes) != len(pending):
            for job, gpu, _ in failures:
                launcher.log(f"FAILURE method={job.label} seed={job.seed} gpu={gpu}")
            launcher.log("ABORT aggregate not executed")
            return 1
        launcher.log("AGGREGATE start")
        result = subprocess.run(
            aggregate_command(python_bin, jobs, results_dir),
            check=False,
        )
        if result.returncode != 0:
            launcher.log(f"AGGREGATE failed exit={result.returncode}")
            return result.returncode
        launcher.log("AGGREGATE success")
        launcher.log("COMPLETE all confirmatory jobs")
        return 0
    except KeyboardInterrupt:
        launcher.log("INTERRUPTED cleaning child processes")
        launcher.terminate_children()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

