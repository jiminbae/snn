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
    "Dataset",
    "Model",
    "Accuracy",
    "Soft Accuracy",
    "Hard Accuracy",
    "Effective Timestep",
    "Hard Effective Timestep",
    "Executed Timestep",
    "Layer1 Hard Timestep",
    "Layer2 Hard Timestep",
    "Raw Spike Rate",
    "Gated Spike Rate",
    "Prefix Spike Rate",
    "Energy Proxy",
    "Prefix Energy Proxy",
    "Loop Energy Proxy",
    "Hard Budget Proxy",
    "Hard Prefix Steps",
    "Hard Prefix Masks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChronoSkip tradeoffs on event-frame datasets.")
    parser.add_argument("--dataset", choices=["nmnist", "dvs_gesture"], default="nmnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/event_chronoskip_tradeoff")
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--event-frame-mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--event-downsample-size", type=int, default=None)
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.05)
    parser.add_argument("--hard-budget-sharpness", type=float, default=5.0)
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


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def row_from_summary(label: str, model: str, dataset: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "Run": label,
        "Dataset": dataset,
        "Model": model,
        "Accuracy": summary.get("test_accuracy", summary.get("test_acc", 0.0)),
        "Soft Accuracy": summary.get("soft_acc", 0.0),
        "Hard Accuracy": summary.get("hard_acc", 0.0),
        "Effective Timestep": summary.get("effective_timestep", 0.0),
        "Hard Effective Timestep": summary.get("hard_effective_timestep", 0.0),
        "Executed Timestep": summary.get("executed_timestep", 0.0),
        "Layer1 Hard Timestep": summary.get("layer1_hard_timestep", 0.0),
        "Layer2 Hard Timestep": summary.get("layer2_hard_timestep", 0.0),
        "Raw Spike Rate": summary.get("raw_spike_rate", 0.0),
        "Gated Spike Rate": summary.get("gated_spike_rate", 0.0),
        "Prefix Spike Rate": summary.get("prefix_spike_rate", 0.0),
        "Energy Proxy": summary.get("energy_proxy", 0.0),
        "Prefix Energy Proxy": summary.get("prefix_energy_proxy", 0.0),
        "Loop Energy Proxy": summary.get("loop_energy_proxy", 0.0),
        "Hard Budget Proxy": hard_budget_proxy_value(summary),
        "Hard Prefix Steps": compact_json(summary.get("hard_prefix_steps", {})),
        "Hard Prefix Masks": compact_json(summary.get("hard_prefix_masks", [])),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite = [
        ("fixed_lif_T8", "fixed_lif", 8, {}),
        ("fixed_lif_T6", "fixed_lif", 6, {}),
        ("fixed_lif_T4", "fixed_lif", 4, {}),
        ("fixed_lif_T3", "fixed_lif", 3, {}),
        ("global_chronoskip_s2h_T6target", "global_chronoskip_s2h", args.tmax, {"gate_init": 2.5, "target_timestep": 6, "lambda_hard_budget": 0.05}),
        (
            "layerwise_chronoskip_s2h_dep_T6target",
            "layerwise_chronoskip_s2h",
            args.tmax,
            {"gate_init": 2.5, "target_timestep": 6, "lambda_hard_budget": 0.05, "dependency_constrained_prefix": True},
        ),
        ("global_chronoskip_s2h_T4target", "global_chronoskip_s2h", args.tmax, {"gate_init": 2.0, "target_timestep": 4, "lambda_hard_budget": 0.05}),
        (
            "layerwise_chronoskip_s2h_dep_T4target",
            "layerwise_chronoskip_s2h",
            args.tmax,
            {"gate_init": 2.0, "target_timestep": 4, "lambda_hard_budget": 0.05, "dependency_constrained_prefix": True},
        ),
    ]
    rows: list[dict[str, Any]] = []

    for label, model_name, tmax, config in suite:
        run_name = f"event_{timestamp}_{label}"
        cmd = [
            sys.executable,
            str(root / "train.py"),
            "--model", model_name,
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--tmax", str(tmax),
            "--device", args.device,
            "--seed", str(args.seed),
            "--data-dir", args.data_dir,
            "--results-dir", args.results_dir,
            "--run-name", run_name,
            "--event-frame-mode", args.event_frame_mode,
            "--gate-init", str(config.get("gate_init", 2.5)),
            "--lambda-spike", str(args.lambda_spike),
            "--eta-time", str(args.eta_time),
            "--lambda-hard-budget", str(config.get("lambda_hard_budget", 0.0)),
            "--hard-budget-sharpness", str(args.hard_budget_sharpness),
            "--target-timestep", str(config.get("target_timestep", 0.0)),
            "--target-budget-weight", str(args.target_budget_weight),
            "--min-prefix-steps", str(args.min_prefix_steps),
            "--gate-threshold", str(args.gate_threshold),
            "--hard-prefix-eval",
            "--hard-prefix-unscaled",
        ]
        if args.event_downsample_size is not None:
            cmd.extend(["--event-downsample-size", str(args.event_downsample_size)])
        elif args.dataset == "dvs_gesture":
            cmd.extend(["--event-downsample-size", "64"])
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
        rows.append(row_from_summary(label, model_name, args.dataset, summary))

    comparison_path = root / args.results_dir / "comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved event ChronoSkip tradeoff table to {comparison_path}")


if __name__ == "__main__":
    main()
