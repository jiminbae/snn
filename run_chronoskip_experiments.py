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
    "Model",
    "Accuracy",
    "Soft Accuracy",
    "Hard Accuracy",
    "Raw Spike Rate",
    "Gated Spike Rate",
    "Prefix Spike Rate",
    "Effective Timestep",
    "Hard Effective Timestep",
    "Executed Timestep",
    "Layer1 Effective Timestep",
    "Layer2 Effective Timestep",
    "Layer1 Hard Timestep",
    "Layer2 Hard Timestep",
    "Energy Proxy",
    "Prefix Energy Proxy",
    "Loop Energy Proxy",
    "Hard Budget Proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ChronoSkip experiment suite.")
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar10", "nmnist", "dvs_gesture"], default="fashionmnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--event-frame-mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--event-downsample-size", type=int, default=None)
    parser.add_argument("--gate-init", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.02)
    parser.add_argument("--lambda-hard-budget", type=float, default=0.0)
    parser.add_argument("--hard-budget-sharpness", type=float, default=20.0)
    parser.add_argument("--target-timestep", type=float, default=0.0)
    parser.add_argument("--target-budget-weight", type=float, default=0.0)
    parser.add_argument("--target-budget-mode", choices=["upper", "two_sided", "l2"], default="upper")
    parser.add_argument("--min-target-timestep", type=float, default=0.0)
    parser.add_argument("--min-target-weight", type=float, default=0.0)
    parser.add_argument("--spike-cost-mode", choices=["raw", "gated", "mixed"], default="gated")
    parser.add_argument("--hard-prefix-eval", action="store_true")
    parser.add_argument("--hard-prefix-unscaled", action="store_true")
    parser.add_argument("--dependency-constrained-prefix", action="store_true")
    parser.add_argument("--min-prefix-steps", type=int, default=1)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--hard-ce-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--reg-warmup-epochs", type=int, default=5)
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


def row_from_summary(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": label,
        "Accuracy": summary.get("test_accuracy", summary.get("test_acc", 0.0)),
        "Soft Accuracy": summary.get("soft_acc", 0.0),
        "Hard Accuracy": summary.get("hard_acc", 0.0),
        "Raw Spike Rate": summary.get("raw_spike_rate", 0.0),
        "Gated Spike Rate": summary.get("gated_spike_rate", 0.0),
        "Prefix Spike Rate": summary.get("prefix_spike_rate", 0.0),
        "Effective Timestep": summary.get("effective_timestep", 0.0),
        "Hard Effective Timestep": summary.get("hard_effective_timestep", 0.0),
        "Executed Timestep": summary.get("executed_timestep", 0.0),
        "Layer1 Effective Timestep": summary.get("layer1_effective_timestep", 0.0),
        "Layer2 Effective Timestep": summary.get("layer2_effective_timestep", 0.0),
        "Layer1 Hard Timestep": summary.get("layer1_hard_timestep", 0.0),
        "Layer2 Hard Timestep": summary.get("layer2_hard_timestep", 0.0),
        "Energy Proxy": summary.get("energy_proxy", 0.0),
        "Prefix Energy Proxy": summary.get("prefix_energy_proxy", 0.0),
        "Loop Energy Proxy": summary.get("loop_energy_proxy", 0.0),
        "Hard Budget Proxy": hard_budget_proxy_value(summary),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite = [
        ("fixed_lif_T8", "fixed_lif", 8),
        ("fixed_lif_T6", "fixed_lif", 6),
        ("fixed_lif_T4", "fixed_lif", 4),
        ("fixed_lif_T2", "fixed_lif", 2),
        ("soft_gate_T8", "soft_gate", 8),
        ("global_chronoskip_T8", "global_chronoskip", 8),
        ("global_chronoskip_s2h_T8", "global_chronoskip_s2h", 8),
        ("layerwise_chronoskip_T8", "layerwise_chronoskip", 8),
        ("layerwise_chronoskip_s2h_T8", "layerwise_chronoskip_s2h", 8),
    ]
    rows: list[dict[str, Any]] = []

    for label, model_name, tmax in suite:
        run_name = f"chronoskip_{timestamp}_{label}"
        cmd = [
            sys.executable,
            str(root / "train.py"),
            "--model",
            model_name,
            "--dataset",
            args.dataset,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--tmax",
            str(tmax),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--results-dir",
            args.results_dir,
            "--data-dir",
            args.data_dir,
            "--run-name",
            run_name,
            "--event-frame-mode",
            args.event_frame_mode,
            "--gate-init",
            str(args.gate_init),
            "--lambda-spike",
            str(args.lambda_spike),
            "--eta-time",
            str(args.eta_time),
            "--lambda-hard-budget",
            str(args.lambda_hard_budget),
            "--hard-budget-sharpness",
            str(args.hard_budget_sharpness),
            "--target-timestep",
            str(args.target_timestep),
            "--target-budget-weight",
            str(args.target_budget_weight),
            "--target-budget-mode",
            args.target_budget_mode,
            "--min-target-timestep",
            str(args.min_target_timestep),
            "--min-target-weight",
            str(args.min_target_weight),
            "--spike-cost-mode",
            args.spike_cost_mode,
            "--min-prefix-steps",
            str(args.min_prefix_steps),
            "--gate-threshold",
            str(args.gate_threshold),
            "--hard-ce-weight",
            str(args.hard_ce_weight),
            "--consistency-weight",
            str(args.consistency_weight),
            "--reg-warmup-epochs",
            str(args.reg_warmup_epochs),
        ]
        if args.event_downsample_size is not None:
            cmd.extend(["--event-downsample-size", str(args.event_downsample_size)])
        elif args.dataset == "dvs_gesture":
            cmd.extend(["--event-downsample-size", "64"])
        if args.amp:
            cmd.append("--amp")
        if args.hard_prefix_eval:
            cmd.append("--hard-prefix-eval")
        if args.hard_prefix_unscaled:
            cmd.append("--hard-prefix-unscaled")
        if args.dependency_constrained_prefix:
            cmd.append("--dependency-constrained-prefix")
        if args.limit_train_batches is not None:
            cmd.extend(["--limit-train-batches", str(args.limit_train_batches)])
        if args.limit_test_batches is not None:
            cmd.extend(["--limit-test-batches", str(args.limit_test_batches)])

        print(f"\nRunning {label}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=root, check=True)
        summary = load_summary(root / args.results_dir / run_name / "summary.json")
        rows.append(row_from_summary(label, summary))

    comparison_path = root / args.results_dir / "comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved comparison table to {comparison_path}")
    for row in rows:
        print(
            f"{row['Model']:>28} | acc {float(row['Accuracy']):6.2f}% | "
            f"soft {float(row['Soft Accuracy']):6.2f}% | hard {float(row['Hard Accuracy']):6.2f}% | "
            f"raw {float(row['Raw Spike Rate']):.5f} | gated {float(row['Gated Spike Rate']):.5f} | "
            f"prefix {float(row['Prefix Spike Rate']):.5f} | T {float(row['Effective Timestep']):.2f}/hard {float(row['Hard Effective Timestep']):.2f}/exec {float(row['Executed Timestep']):.2f} | "
            f"energy proxy {float(row['Energy Proxy']):.5f} | loop proxy {float(row['Loop Energy Proxy']):.5f} | "
            f"hard budget {float(row['Hard Budget Proxy']):.2f}"
        )


if __name__ == "__main__":
    main()
