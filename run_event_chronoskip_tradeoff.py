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
    "Baseline Type",
    "Base Tmax",
    "Temporal Prefix Mode",
    "Temporal Prefix Steps",
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
    parser.add_argument("--base-tmax", type=int, default=8)
    parser.add_argument("--tmax", type=int, default=None, help="Alias for --base-tmax.")
    parser.add_argument("--event-frame-mode", choices=["binary", "count"], default="binary")
    parser.add_argument("--event-downsample-size", type=int, default=None)
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.05)
    parser.add_argument("--hard-budget-sharpness", type=float, default=5.0)
    parser.add_argument("--target-budget-weight", type=float, default=0.05)
    parser.add_argument("--target-budget-mode", choices=["upper", "two_sided", "l2"], default="upper")
    parser.add_argument("--min-target-timestep", type=float, default=0.0)
    parser.add_argument("--min-target-weight", type=float, default=0.0)
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


def prefix_steps_value(config: dict[str, Any], summary: dict[str, Any]) -> Any:
    steps = summary.get("hard_prefix_steps", {})
    if steps:
        return steps
    if config["baseline_type"] == "fixed_prefix":
        value = float(config.get("temporal_prefix_steps", 0))
        return {"global": value}
    if config["baseline_type"] == "fixed_rebin":
        value = float(config.get("tmax", 0))
        return {"global": value}
    return {}


def prefix_masks_value(config: dict[str, Any], summary: dict[str, Any]) -> Any:
    masks = summary.get("hard_prefix_masks", [])
    if masks:
        return masks
    tmax = int(config.get("tmax", 0))
    if config["baseline_type"] == "fixed_prefix":
        steps = int(config.get("temporal_prefix_steps", 0))
        return [1.0 if idx < steps else 0.0 for idx in range(tmax)]
    if config["baseline_type"] == "fixed_rebin":
        return [1.0 for _ in range(tmax)]
    return []


def row_from_summary(label: str, config: dict[str, Any], dataset: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "Run": label,
        "Dataset": dataset,
        "Model": config["model"],
        "Baseline Type": config["baseline_type"],
        "Base Tmax": config["base_tmax"],
        "Temporal Prefix Mode": config.get("temporal_prefix_mode", "none"),
        "Temporal Prefix Steps": config.get("temporal_prefix_steps", 0),
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
        "Hard Prefix Steps": compact_json(prefix_steps_value(config, summary)),
        "Hard Prefix Masks": compact_json(prefix_masks_value(config, summary)),
    }


def build_suite(base_tmax: int) -> list[tuple[str, dict[str, Any]]]:
    suite: list[tuple[str, dict[str, Any]]] = []
    for t in (8, 6, 4, 3):
        suite.append(
            (
                f"fixed_rebin_T{t}",
                {
                    "model": "fixed_lif",
                    "baseline_type": "fixed_rebin",
                    "tmax": t,
                    "base_tmax": t,
                    "temporal_prefix_mode": "none",
                    "temporal_prefix_steps": 0,
                },
            )
        )
    for t in (6, 4, 3, 2, 1):
        suite.append(
            (
                f"fixed_prefix_T{t}",
                {
                    "model": "fixed_lif",
                    "baseline_type": "fixed_prefix",
                    "tmax": base_tmax,
                    "base_tmax": base_tmax,
                    "temporal_prefix_mode": "truncate",
                    "temporal_prefix_steps": t,
                },
            )
        )
    suite.extend(
        [
            (
                "global_chronoskip_s2h_T6target",
                {
                    "model": "global_chronoskip_s2h",
                    "baseline_type": "chronoskip",
                    "tmax": base_tmax,
                    "base_tmax": base_tmax,
                    "gate_init": 2.5,
                    "target_timestep": 6,
                    "lambda_hard_budget": 0.05,
                    "temporal_prefix_mode": "none",
                    "temporal_prefix_steps": 0,
                },
            ),
            (
                "layerwise_chronoskip_s2h_dep_T6target",
                {
                    "model": "layerwise_chronoskip_s2h",
                    "baseline_type": "chronoskip",
                    "tmax": base_tmax,
                    "base_tmax": base_tmax,
                    "gate_init": 2.5,
                    "target_timestep": 6,
                    "lambda_hard_budget": 0.05,
                    "dependency_constrained_prefix": True,
                    "temporal_prefix_mode": "none",
                    "temporal_prefix_steps": 0,
                },
            ),
            (
                "global_chronoskip_s2h_T4target",
                {
                    "model": "global_chronoskip_s2h",
                    "baseline_type": "chronoskip",
                    "tmax": base_tmax,
                    "base_tmax": base_tmax,
                    "gate_init": 2.0,
                    "target_timestep": 4,
                    "lambda_hard_budget": 0.05,
                    "temporal_prefix_mode": "none",
                    "temporal_prefix_steps": 0,
                },
            ),
            (
                "layerwise_chronoskip_s2h_dep_T4target",
                {
                    "model": "layerwise_chronoskip_s2h",
                    "baseline_type": "chronoskip",
                    "tmax": base_tmax,
                    "base_tmax": base_tmax,
                    "gate_init": 2.0,
                    "target_timestep": 4,
                    "lambda_hard_budget": 0.05,
                    "dependency_constrained_prefix": True,
                    "temporal_prefix_mode": "none",
                    "temporal_prefix_steps": 0,
                },
            ),
        ]
    )
    return suite


def main() -> None:
    args = parse_args()
    base_tmax = args.base_tmax if args.tmax is None else args.tmax
    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []

    for label, config in build_suite(base_tmax):
        run_name = f"event_{timestamp}_{label}"
        cmd = [
            sys.executable,
            str(root / "train.py"),
            "--model", config["model"],
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--tmax", str(config["tmax"]),
            "--device", args.device,
            "--seed", str(args.seed),
            "--data-dir", args.data_dir,
            "--results-dir", args.results_dir,
            "--run-name", run_name,
            "--event-frame-mode", args.event_frame_mode,
            "--temporal-prefix-mode", config.get("temporal_prefix_mode", "none"),
            "--temporal-prefix-steps", str(config.get("temporal_prefix_steps", 0)),
            "--gate-init", str(config.get("gate_init", 2.5)),
            "--lambda-spike", str(args.lambda_spike),
            "--eta-time", str(args.eta_time),
            "--lambda-hard-budget", str(config.get("lambda_hard_budget", 0.0)),
            "--hard-budget-sharpness", str(args.hard_budget_sharpness),
            "--target-timestep", str(config.get("target_timestep", 0.0)),
            "--target-budget-weight", str(args.target_budget_weight),
            "--target-budget-mode", args.target_budget_mode,
            "--min-target-timestep", str(args.min_target_timestep),
            "--min-target-weight", str(args.min_target_weight),
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
        rows.append(row_from_summary(label, config, args.dataset, summary))

    comparison_path = root / args.results_dir / "comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved event ChronoSkip tradeoff table to {comparison_path}")


if __name__ == "__main__":
    main()
