from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "Run",
    "Model",
    "Lambda Hard Budget",
    "Dependency Constrained Prefix",
    "Accuracy",
    "Soft Accuracy",
    "Hard Accuracy",
    "Effective Timestep",
    "Hard Effective Timestep",
    "Executed Timestep",
    "Layer1 Hard Timestep",
    "Layer2 Hard Timestep",
    "Prefix Spike Rate",
    "Prefix Energy Proxy",
    "Loop Energy Proxy",
    "Hard Budget Proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChronoSkip hard-budget diagnostics.")
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar10"], default="fashionmnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/diagnostics_hard_budget")
    parser.add_argument("--gate-init", type=float, default=4.0)
    parser.add_argument("--eta-time", type=float, default=0.05)
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--hard-budget-sharpness", type=float, default=20.0)
    parser.add_argument("--target-timestep", type=float, default=6.0)
    parser.add_argument("--target-budget-weight", type=float, default=0.05)
    parser.add_argument("--min-prefix-steps", type=int, default=1)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hard_budget_proxy_value(summary: dict[str, Any]) -> float:
    value = summary.get("hard_budget_proxy", 0.0)
    if isinstance(value, dict):
        for key in ("average", "global", "layer1"):
            if key in value:
                return float(value[key])
        return 0.0
    return float(value)


def row_from_summary(label: str, config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "Run": label,
        "Model": config["model"],
        "Lambda Hard Budget": config["lambda_hard_budget"],
        "Dependency Constrained Prefix": config.get("dependency_constrained_prefix", False),
        "Accuracy": summary.get("test_accuracy", summary.get("test_acc", 0.0)),
        "Soft Accuracy": summary.get("soft_acc", 0.0),
        "Hard Accuracy": summary.get("hard_acc", 0.0),
        "Effective Timestep": summary.get("effective_timestep", 0.0),
        "Hard Effective Timestep": summary.get("hard_effective_timestep", 0.0),
        "Executed Timestep": summary.get("executed_timestep", 0.0),
        "Layer1 Hard Timestep": summary.get("layer1_hard_timestep", 0.0),
        "Layer2 Hard Timestep": summary.get("layer2_hard_timestep", 0.0),
        "Prefix Spike Rate": summary.get("prefix_spike_rate", 0.0),
        "Prefix Energy Proxy": summary.get("prefix_energy_proxy", 0.0),
        "Loop Energy Proxy": summary.get("loop_energy_proxy", 0.0),
        "Hard Budget Proxy": hard_budget_proxy_value(summary),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments = [
        ("global_hb_002", {"model": "global_chronoskip_s2h", "lambda_hard_budget": 0.02}),
        ("global_hb_005", {"model": "global_chronoskip_s2h", "lambda_hard_budget": 0.05}),
        ("global_hb_010", {"model": "global_chronoskip_s2h", "lambda_hard_budget": 0.10}),
        ("layerwise_hb_005", {"model": "layerwise_chronoskip_s2h", "lambda_hard_budget": 0.05}),
        ("layerwise_hb_010", {"model": "layerwise_chronoskip_s2h", "lambda_hard_budget": 0.10}),
        (
            "layerwise_hb_005_dep",
            {"model": "layerwise_chronoskip_s2h", "lambda_hard_budget": 0.05, "dependency_constrained_prefix": True},
        ),
    ]
    rows: list[dict[str, Any]] = []

    for label, config in experiments:
        run_name = f"diagnostic_{timestamp}_{label}"
        cmd = [
            sys.executable,
            str(root / "train.py"),
            "--model",
            config["model"],
            "--dataset",
            args.dataset,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--data-dir",
            args.data_dir,
            "--results-dir",
            args.results_dir,
            "--run-name",
            run_name,
            "--gate-init",
            str(args.gate_init),
            "--eta-time",
            str(args.eta_time),
            "--lambda-spike",
            str(args.lambda_spike),
            "--lambda-hard-budget",
            str(config["lambda_hard_budget"]),
            "--hard-budget-sharpness",
            str(args.hard_budget_sharpness),
            "--target-timestep",
            str(args.target_timestep),
            "--target-budget-weight",
            str(args.target_budget_weight),
            "--min-prefix-steps",
            str(args.min_prefix_steps),
            "--gate-threshold",
            str(args.gate_threshold),
            "--hard-prefix-eval",
            "--hard-prefix-unscaled",
        ]
        if args.amp:
            cmd.append("--amp")
        if config.get("dependency_constrained_prefix", False):
            cmd.append("--dependency-constrained-prefix")
        if args.limit_train_batches is not None:
            cmd.extend(["--limit-train-batches", str(args.limit_train_batches)])
        if args.limit_test_batches is not None:
            cmd.extend(["--limit-test-batches", str(args.limit_test_batches)])

        print(f"\nRunning {label}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=root, check=True)
        summary = load_summary(root / args.results_dir / run_name / "summary.json")
        rows.append(row_from_summary(label, config, summary))

    comparison_path = root / args.results_dir / "comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved hard-budget diagnostic comparison to {comparison_path}")


if __name__ == "__main__":
    main()
