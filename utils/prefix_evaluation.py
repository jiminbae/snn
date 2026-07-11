"""Dataset-level evaluation and persistence for temporal-prefix diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .logging import save_json
from .plotting import plot_prefix_accuracy_curve, plot_prefix_regression_curve
from .prefix_metrics import (
    consecutive_regression_rate,
    ever_regressed_rate,
    first_correct_timestep,
    mean_negative_temporal_gain,
    negative_temporal_gain,
    prefix_accuracy_auc,
    prefix_accuracy_curve,
    stable_correct_timestep,
    worst_prefix_accuracy,
)


@torch.no_grad()
def evaluate_prefix_diagnostics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: Any,
) -> dict[str, Any]:
    """Aggregate prefix diagnostics over samples, never over batch percentages."""
    model.eval()
    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []
    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="prefix diagnostics", leave=False), start=1):
        if args.limit_test_batches and batch_idx > args.limit_test_batches:
            break
        images = images.to(device, non_blocking=True)
        output = model(
            images,
            mode="soft",
            gate_threshold=args.gate_threshold,
            min_prefix_steps=args.min_prefix_steps,
            temporal_prefix_steps=args.temporal_prefix_steps,
            temporal_prefix_mode=args.temporal_prefix_mode,
            return_prefix_logits=True,
        )
        prefix_logits = output["prefix_logits"]
        final_logits = output["logits"]
        assert isinstance(prefix_logits, Tensor)
        assert isinstance(final_logits, Tensor)
        if not torch.allclose(final_logits, prefix_logits[:, -1], atol=1e-6):
            raise AssertionError("Final logits must equal the final prefix logits.")
        all_logits.append(prefix_logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if not all_logits:
        raise RuntimeError("Prefix diagnostics received no evaluation batches.")
    prefix_logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    curve = prefix_accuracy_curve(prefix_logits, targets)
    regressions = consecutive_regression_rate(prefix_logits, targets)
    first = first_correct_timestep(prefix_logits, targets)
    stable = stable_correct_timestep(prefix_logits, targets)
    sentinel = prefix_logits.shape[1] + 1
    first_valid = first < sentinel
    stable_valid = stable < sentinel

    result: dict[str, Any] = {
        "prefix_accuracy_curve": curve.tolist(),
        "negative_temporal_gain": float(negative_temporal_gain(curve).item()),
        "mean_negative_temporal_gain": float(mean_negative_temporal_gain(curve).item()),
        "consecutive_regression_rate": float(regressions["mean"].item()),
        "consecutive_regression_rate_per_transition": regressions["per_transition"].tolist(),
        "ever_regressed_rate": float(ever_regressed_rate(prefix_logits, targets).item()),
        "worst_prefix_accuracy": float(worst_prefix_accuracy(curve).item()),
        "prefix_accuracy_auc": float(prefix_accuracy_auc(curve).item()),
        "mean_first_correct_timestep": float(first[first_valid].float().mean().item()) if first_valid.any() else 0.0,
        "mean_stable_correct_timestep": float(stable[stable_valid].float().mean().item()) if stable_valid.any() else 0.0,
        "first_correct_valid_fraction": float(first_valid.float().mean().item() * 100.0),
        "stable_correct_valid_fraction": float(stable_valid.float().mean().item() * 100.0),
        "never_correct_fraction": float((~first_valid).float().mean().item() * 100.0),
        "never_stable_fraction": float((~stable_valid).float().mean().item() * 100.0),
        "num_samples": int(targets.shape[0]),
    }
    result.update({f"prefix_accuracy_t{t}": float(value) for t, value in enumerate(curve.tolist(), start=1)})
    return result


def save_prefix_diagnostics(run_dir: str | Path, metrics: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "prefix_metrics.json", metrics)

    curve = [float(value) for value in metrics["prefix_accuracy_curve"]]
    with (run_dir / "prefix_accuracy_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestep", "Accuracy"])
        writer.writerows((t, value) for t, value in enumerate(curve, start=1))

    regressions = [float(value) for value in metrics["consecutive_regression_rate_per_transition"]]
    with (run_dir / "prefix_regression_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["From Timestep", "To Timestep", "Regression Rate"])
        writer.writerows((t, t + 1, value) for t, value in enumerate(regressions, start=1))

    plots_dir = run_dir / "plots"
    plot_prefix_accuracy_curve(curve, plots_dir / "prefix_accuracy_curve.png")
    plot_prefix_regression_curve(regressions, plots_dir / "prefix_regression_curve.png")
