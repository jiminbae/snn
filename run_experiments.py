from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SpikeGate pilot experiment suite.")
    parser.add_argument("--dataset", choices=["fashionmnist", "cifar10"], default="fashionmnist")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tmax", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lambda-spike", type=float, default=0.05)
    parser.add_argument("--eta-time", type=float, default=0.02)
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    parser.add_argument("--hard-prefix-eval", action="store_true")
    parser.add_argument("--hard-prefix-unscaled", action="store_true")
    parser.add_argument("--reg-warmup-epochs", type=int, default=5)
    parser.add_argument("--spike-cost-mode", choices=["gated", "raw", "mixed"], default="gated")
    parser.add_argument("--monotonic-gate", dest="monotonic_gate", action="store_true", default=True)
    parser.add_argument("--no-monotonic-gate", dest="monotonic_gate", action="store_false")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models = ["fixed_lif", "gate_only", "neuron_only", "spikegate"]
    rows: list[dict[str, Any]] = []

    for model_name in models:
        run_name = f"suite_{timestamp}_{model_name}"
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
            str(args.tmax),
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
            "--lambda-spike",
            str(args.lambda_spike),
            "--eta-time",
            str(args.eta_time),
            "--gumbel-tau",
            str(args.gumbel_tau),
            "--reg-warmup-epochs",
            str(args.reg_warmup_epochs),
            "--spike-cost-mode",
            args.spike_cost_mode,
        ]
        if args.amp:
            cmd.append("--amp")
        if args.hard_prefix_eval:
            cmd.append("--hard-prefix-eval")
        if args.hard_prefix_unscaled:
            cmd.append("--hard-prefix-unscaled")
        if args.monotonic_gate and model_name in {"gate_only", "spikegate"}:
            cmd.append("--monotonic-gate")
        if args.limit_train_batches is not None:
            cmd.extend(["--limit-train-batches", str(args.limit_train_batches)])
        if args.limit_test_batches is not None:
            cmd.extend(["--limit-test-batches", str(args.limit_test_batches)])

        print(f"\nRunning {model_name}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=root, check=True)

        summary = load_summary(root / args.results_dir / run_name / "summary.json")
        rows.append(
            {
                "Model": model_name,
                "Accuracy": summary["test_accuracy"],
                "Raw Spike Rate": summary["raw_spike_rate"],
                "Gated Spike Rate": summary["gated_spike_rate"],
                "Prefix Spike Rate": summary["prefix_spike_rate"],
                "Effective Timestep": summary["effective_timestep"],
                "Hard Effective Timestep": summary["hard_effective_timestep"],
                "Energy Proxy": summary["energy_proxy"],
                "Prefix Energy Proxy": summary["prefix_energy_proxy"],
                "Selected Neurons": "; ".join(summary["selected_names"]),
            }
        )

    comparison_path = root / args.results_dir / "comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Model",
                "Accuracy",
                "Raw Spike Rate",
                "Gated Spike Rate",
                "Prefix Spike Rate",
                "Effective Timestep",
                "Hard Effective Timestep",
                "Energy Proxy",
                "Prefix Energy Proxy",
                "Selected Neurons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved comparison table to {comparison_path}")
    for row in rows:
        print(
            f"{row['Model']:>16} | acc {float(row['Accuracy']):6.2f}% | "
            f"raw {float(row['Raw Spike Rate']):.5f} | gated {float(row['Gated Spike Rate']):.5f} | "
            f"prefix {float(row['Prefix Spike Rate']):.5f} | T {float(row['Effective Timestep']):.3f}/hard {float(row['Hard Effective Timestep']):.0f} | "
            f"energy proxy {float(row['Energy Proxy']):.5f} | prefix energy {float(row['Prefix Energy Proxy']):.5f} | "
            f"{row['Selected Neurons']}"
        )


if __name__ == "__main__":
    main()
