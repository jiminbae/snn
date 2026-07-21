#!/usr/bin/env python3
"""Export per-sample prefix trajectories from fixed confirmatory checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.data import _build_datasets, get_dataset_spec
from utils.trajectory_export import load_torch_compat


REQUIRED_RUN_FILES = (
    "config.json",
    "best_checkpoint.pt",
    "temporal_reliability_summary.json",
    "prefix_metrics.json",
)
EXPORT_FORMAT_VERSION = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints(run_dir: Path) -> dict[str, str]:
    return {name: sha256_file(run_dir / name) for name in REQUIRED_RUN_FILES}


def build_trajectory(
    prefix_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    method: str,
    seed: int,
    checkpoint_path: str,
    config: dict[str, Any],
    fingerprints: dict[str, str] | None = None,
    export_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probabilities = prefix_logits.float().softmax(dim=-1)
    top2 = probabilities.topk(k=2, dim=-1).values
    predictions = prefix_logits.argmax(dim=-1)
    true_probability = probabilities.gather(
        2, targets[:, None, None].expand(-1, prefix_logits.shape[1], 1)
    ).squeeze(-1)
    return {
        "prefix_logits": prefix_logits.float(),
        "targets": targets.long(),
        "predictions": predictions.long(),
        "correct": predictions.eq(targets[:, None]),
        "true_class_probability": true_probability,
        "top1_confidence": top2[..., 0],
        "top1_margin": top2[..., 0] - top2[..., 1],
        "sample_index": torch.arange(targets.shape[0], dtype=torch.long),
        "method": method,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "export_format_version": EXPORT_FORMAT_VERSION,
        "source_fingerprints": dict(fingerprints or {}),
        "export_settings": dict(export_settings or {}),
    }


def validate_trajectory(
    trajectory: dict[str, Any],
    *,
    expected_samples: int,
    expected_timesteps: int,
    expected_classes: int,
    expected_final_accuracy: float,
    expected_prefix_curve: list[float],
) -> None:
    logits = trajectory["prefix_logits"]
    targets = trajectory["targets"]
    predictions = trajectory["predictions"]
    correct = trajectory["correct"]
    expected_logits = (expected_samples, expected_timesteps, expected_classes)
    expected_temporal = (expected_samples, expected_timesteps)
    shapes = {
        "prefix_logits": (tuple(logits.shape), expected_logits),
        "targets": (tuple(targets.shape), (expected_samples,)),
        "predictions": (tuple(predictions.shape), expected_temporal),
        "correct": (tuple(correct.shape), expected_temporal),
    }
    for key, (actual, expected) in shapes.items():
        if actual != expected:
            raise ValueError(f"{key} shape expected {expected}, actual {actual}")
    if not torch.equal(predictions, logits.argmax(dim=-1)):
        raise ValueError("predictions do not match prefix_logits")
    if not torch.equal(correct, predictions.eq(targets[:, None])):
        raise ValueError("correct mask does not match predictions and targets")
    final_accuracy = float(correct[:, -1].float().mean().item() * 100.0)
    if abs(final_accuracy - expected_final_accuracy) > 1e-4:
        raise ValueError(
            f"final accuracy expected {expected_final_accuracy}, actual {final_accuracy}"
        )
    curve = correct.float().mean(dim=0) * 100.0
    expected_curve = torch.tensor(expected_prefix_curve, dtype=curve.dtype)
    if not torch.allclose(curve, expected_curve, atol=1e-4, rtol=0.0):
        raise ValueError(
            f"prefix curve expected {expected_prefix_curve}, actual {curve.tolist()}"
        )
    indices = trajectory["sample_index"]
    if not torch.equal(indices, torch.arange(expected_samples)):
        raise ValueError("sample_index is not the official deterministic test order")


def validate_alignment(trajectories: list[dict[str, Any]]) -> None:
    if not trajectories:
        raise ValueError("No trajectories supplied for alignment")
    reference = trajectories[0]
    for trajectory in trajectories[1:]:
        for key in ("sample_index", "targets"):
            if not torch.equal(reference[key], trajectory[key]):
                raise ValueError(
                    f"Trajectory alignment failed for {key}: "
                    f"{reference['method']} seed {reference['seed']} vs "
                    f"{trajectory['method']} seed {trajectory['seed']}"
                )
        if tuple(reference["prefix_logits"].shape) != tuple(
            trajectory["prefix_logits"].shape
        ):
            raise ValueError("Trajectory alignment failed for prefix shape")


@torch.no_grad()
def export_run(
    run_dir: Path,
    output_path: Path,
    *,
    device_name: str,
    batch_size: int | None,
    num_workers: int,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing required files {missing}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cudnn.benchmark = True
    summary = json.loads(
        (run_dir / "temporal_reliability_summary.json").read_text(encoding="utf-8")
    )
    prefix_metrics = json.loads(
        (run_dir / "prefix_metrics.json").read_text(encoding="utf-8")
    )
    if config["dataset"] != "nmnist" or config["model"] != "fixed_lif":
        raise ValueError(f"{run_dir}: expected fixed_lif N-MNIST run")
    _, test_dataset = _build_datasets(
        config["dataset"],
        config.get("data_dir", "data"),
        tmax=int(config["tmax"]),
        event_frame_mode=config.get("event_frame_mode", "binary"),
        event_downsample_size=config.get("resolved_event_downsample_size"),
    )
    resolved_batch_size = int(config["batch_size"]) if batch_size is None else batch_size
    loader = DataLoader(
        test_dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    device = torch.device(
        device_name
        if not device_name.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    model = build_model(
        config["model"],
        dataset=config["dataset"],
        tmax=int(config["tmax"]),
        gate_init=float(config.get("gate_init", 5.0)),
    ).to(device)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = load_torch_compat(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    all_logits = []
    all_targets = []
    for images, targets in tqdm(loader, desc=f"export {run_dir.name}", leave=False):
        images = images.to(device, non_blocking=True)
        output = model(
            images,
            mode="soft",
            gate_threshold=config.get("gate_threshold", 0.5),
            min_prefix_steps=config.get("min_prefix_steps", 1),
            temporal_prefix_steps=config.get("temporal_prefix_steps", 0),
            temporal_prefix_mode=config.get("temporal_prefix_mode", "none"),
            return_prefix_logits=True,
        )
        all_logits.append(output["prefix_logits"].detach().cpu())
        all_targets.append(targets.cpu())
    method = run_dir.parent.name
    trajectory = build_trajectory(
        torch.cat(all_logits),
        torch.cat(all_targets),
        method=method,
        seed=int(config["seed"]),
        checkpoint_path=str(checkpoint_path),
        config=config,
        fingerprints=source_fingerprints(run_dir),
        export_settings={"batch_size": resolved_batch_size, "cudnn_benchmark": True},
    )
    spec = get_dataset_spec("nmnist")
    validate_trajectory(
        trajectory,
        expected_samples=len(test_dataset),
        expected_timesteps=int(config["tmax"]),
        expected_classes=int(spec["num_classes"]),
        expected_final_accuracy=float(summary["final_accuracy"]),
        expected_prefix_curve=prefix_metrics["prefix_accuracy_curve"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectory, output_path)
    return {
        "output_path": str(output_path),
        "samples": len(test_dataset),
        "method": method,
        "seed": int(config["seed"]),
        "validated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    result = export_run(
        Path(args.run_dir),
        Path(args.output),
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

