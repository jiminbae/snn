"""Plot generation for SpikeGate experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metric_csv(metrics_path: str | Path) -> list[dict[str, float]]:
    with Path(metrics_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed: list[dict[str, float]] = []
    for row in rows:
        parsed.append({key: float(value) for key, value in row.items() if value != ""})
    return parsed


def _line_plot(rows: list[dict[str, float]], key: str, ylabel: str, output: Path) -> None:
    if not rows or key not in rows[-1]:
        return
    epochs = [int(row["epoch"]) for row in rows]
    values = [row[key] for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, values, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_training_curves(metrics_path: str | Path, plots_dir: str | Path) -> None:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_metric_csv(metrics_path)
    _line_plot(rows, "test_acc", "Test accuracy (%)", plots_dir / "accuracy_curve.png")
    _line_plot(rows, "test_spike_rate", "Average spike rate", plots_dir / "spike_rate_curve.png")
    _line_plot(rows, "effective_timestep", "Effective timestep", plots_dir / "effective_timestep_curve.png")
    _line_plot(rows, "energy_proxy", "Energy proxy", plots_dir / "energy_proxy_curve.png")


def plot_timestep_gates(gates: list[float], output: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(gates) + 1), gates)
    plt.xlabel("Timestep")
    plt.ylabel("Gate value")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_candidate_probabilities(
    candidate_names: list[str],
    layer_probs: list[list[float]],
    plots_dir: str | Path,
) -> None:
    plots_dir = Path(plots_dir)
    for layer_idx, probs in enumerate(layer_probs, start=1):
        plt.figure(figsize=(7, 4))
        plt.bar(candidate_names, probs)
        plt.xlabel("Candidate")
        plt.ylabel("Probability")
        plt.ylim(0.0, 1.05)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"candidate_probabilities_layer{layer_idx}.png", dpi=160)
        plt.close()
