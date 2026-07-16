"""Plot generation for ChronoSkip experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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
    _line_plot(rows, "train_loss", "Train loss", plots_dir / "train_loss_curve.png")
    _line_plot(rows, "val_loss", "Validation loss", plots_dir / "val_loss_curve.png")
    _line_plot(rows, "val_acc", "Validation accuracy (%)", plots_dir / "val_accuracy_curve.png")
    _line_plot(rows, "test_acc", "Test accuracy (%)", plots_dir / "accuracy_curve.png")
    _line_plot(rows, "soft_acc", "Soft accuracy (%)", plots_dir / "soft_accuracy_curve.png")
    _line_plot(rows, "hard_acc", "Hard accuracy (%)", plots_dir / "hard_accuracy_curve.png")
    _line_plot(rows, "raw_spike_rate", "Raw spike rate", plots_dir / "raw_spike_rate_curve.png")
    _line_plot(rows, "gated_spike_rate", "Gated spike rate", plots_dir / "gated_spike_rate_curve.png")
    _line_plot(rows, "prefix_spike_rate", "Prefix spike rate", plots_dir / "prefix_spike_rate_curve.png")
    _line_plot(rows, "effective_timestep", "Effective timestep", plots_dir / "effective_timestep_curve.png")
    _line_plot(rows, "hard_effective_timestep", "Hard effective timestep", plots_dir / "hard_effective_timestep_curve.png")
    _line_plot(rows, "executed_timestep", "Executed timestep", plots_dir / "executed_timestep_curve.png")
    _line_plot(rows, "layer1_effective_timestep", "Layer 1 effective timestep", plots_dir / "layer1_effective_timestep_curve.png")
    _line_plot(rows, "layer2_effective_timestep", "Layer 2 effective timestep", plots_dir / "layer2_effective_timestep_curve.png")
    _line_plot(rows, "layer1_hard_timestep", "Layer 1 hard timestep", plots_dir / "layer1_hard_timestep_curve.png")
    _line_plot(rows, "layer2_hard_timestep", "Layer 2 hard timestep", plots_dir / "layer2_hard_timestep_curve.png")
    _line_plot(rows, "train_hard_budget_cost", "Train hard budget cost", plots_dir / "train_hard_budget_cost_curve.png")
    _line_plot(rows, "train_target_budget_loss", "Train target budget loss", plots_dir / "train_target_budget_loss_curve.png")
    _line_plot(rows, "train_min_target_loss", "Train min target loss", plots_dir / "train_min_target_loss_curve.png")
    _line_plot(rows, "train_hard_budget_proxy", "Train hard budget proxy", plots_dir / "train_hard_budget_proxy_curve.png")
    _line_plot(rows, "energy_proxy", "Energy proxy", plots_dir / "energy_proxy_curve.png")
    _line_plot(rows, "prefix_energy_proxy", "Prefix energy proxy", plots_dir / "prefix_energy_proxy_curve.png")
    _line_plot(rows, "loop_energy_proxy", "Loop energy proxy", plots_dir / "loop_energy_proxy_curve.png")


def _plot_gate_values(values: list[float], title: str, output: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(values) + 1), values)
    plt.xlabel("Timestep")
    plt.ylabel("Gate value")
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_timestep_gates(gates: Any, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(gates, dict):
        for layer_name, values in gates.items():
            safe_name = str(layer_name).replace(" ", "_")
            _plot_gate_values([float(v) for v in values], str(layer_name), output.with_name(f"{output.stem}_{safe_name}{output.suffix}"))
        return
    _plot_gate_values([float(v) for v in gates], "Global timestep gate", output)


def plot_prefix_accuracy_curve(values: list[float], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    timesteps = list(range(1, len(values) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(timesteps, values, marker="o", linewidth=2)
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy (%)")
    plt.title("Prefix Accuracy")
    plt.xticks(timesteps)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_prefix_regression_curve(
    population_values: list[float],
    conditional_values: list[float],
    conditional_valid: list[bool],
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    transitions = list(range(1, len(population_values) + 1))
    labels = [f"{t}->{t + 1}" for t in transitions]
    valid_x = [t for t, valid in zip(transitions, conditional_valid) if valid]
    valid_y = [value for value, valid in zip(conditional_values, conditional_valid) if valid]
    invalid_x = [t for t, valid in zip(transitions, conditional_valid) if not valid]
    plt.figure(figsize=(6, 4))
    plt.plot(transitions, population_values, marker="o", linewidth=2, label="Population")
    if valid_x:
        plt.plot(valid_x, valid_y, marker="s", linewidth=2, label="Conditional")
    if invalid_x:
        plt.scatter(invalid_x, [0.0] * len(invalid_x), marker="x", label="Conditional unavailable")
    plt.xlabel("Timestep Transition")
    plt.ylabel("Regression Rate (%)")
    plt.title("Correct-to-Incorrect Prefix Regression")
    plt.xticks(transitions, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
