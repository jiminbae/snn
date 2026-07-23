"""Deterministic trajectory export from a frozen prefix-capable backbone."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .stopping_analysis import validate_trajectory_payload


def load_torch_compat(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def export_trajectory_payload(
    model: nn.Module,
    loader: DataLoader,
    sample_indices: Tensor,
    *,
    split: str,
    device: torch.device,
    forward_args: Any,
    metadata: dict[str, Any],
    include_hidden_features: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_logits, all_targets, all_hidden = [], [], []
    for images, targets in loader:
        output = model(
            images.to(device), mode="soft", gate_threshold=forward_args.gate_threshold,
            min_prefix_steps=forward_args.min_prefix_steps,
            temporal_prefix_steps=forward_args.temporal_prefix_steps,
            temporal_prefix_mode=forward_args.temporal_prefix_mode, return_prefix_logits=True,
            **(
                {"return_temporal_features": True} if include_hidden_features else {}
            ),
        )
        logits, final_logits = output["prefix_logits"], output["logits"]
        if not torch.allclose(final_logits, logits[:, -1], atol=1e-6):
            raise RuntimeError("Final logits do not match the last prefix logits.")
        all_logits.append(logits.detach().cpu().float())
        all_targets.append(targets.detach().cpu().long())
        if include_hidden_features:
            hidden = output.get("temporal_features")
            if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[:2] != logits.shape[:2]:
                raise RuntimeError("Model did not return aligned [B,T,H] temporal_features.")
            if not torch.isfinite(hidden).all():
                raise RuntimeError("temporal_features contain NaN or Inf.")
            all_hidden.append(hidden.detach().cpu().float())
    if not all_logits:
        raise RuntimeError(f"No samples were exported for split '{split}'.")
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probabilities = logits.softmax(dim=-1)
    top2 = probabilities.topk(min(2, probabilities.shape[-1]), dim=-1).values
    predictions = logits.argmax(dim=-1)
    payload = {
        "split": split, "prefix_logits": logits, "targets": targets, "predictions": predictions,
        "confidence": top2[..., 0],
        "entropy": -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1),
        "margin": top2[..., 0] - top2[..., 1] if top2.shape[-1] > 1 else top2[..., 0],
        "correct": predictions.eq(targets[:, None]), "sample_indices": sample_indices.cpu().long(),
        "metadata": metadata,
    }
    if include_hidden_features:
        hidden_features = torch.cat(all_hidden)
        if hidden_features.shape[:2] != logits.shape[:2]:
            raise RuntimeError("Exported hidden features do not align with prefix logits.")
        payload["hidden_features"] = hidden_features
        payload["hidden_feature_metadata"] = metadata.get("hidden_feature_metadata", {
            "format_version": 1, "uses_target": False, "causal": True,
            "dimension": int(hidden_features.shape[-1]),
        })
    n, _, _ = validate_trajectory_payload(payload)
    if payload["sample_indices"].shape != (n,):
        raise ValueError("sample_indices must match exported sample count.")
    return payload


def build_export_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    split_indices: dict[str, Any],
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, DataLoader], dict[str, Tensor]]:
    train_indices = torch.as_tensor(split_indices["train_indices"], dtype=torch.long)
    val_indices = torch.as_tensor(split_indices["val_indices"], dtype=torch.long)
    train_set, val_set = set(train_indices.tolist()), set(val_indices.tolist())
    if train_set & val_set or train_set | val_set != set(range(len(train_dataset))):
        raise ValueError("Saved train/validation indices must be disjoint and cover the training dataset.")
    indices = {"train": train_indices, "val": val_indices, "test": torch.arange(len(test_dataset))}
    datasets = {"train": Subset(train_dataset, train_indices.tolist()),
                "val": Subset(train_dataset, val_indices.tolist()), "test": test_dataset}
    kwargs = {"batch_size": batch_size, "shuffle": False, "drop_last": False,
              "num_workers": num_workers, "persistent_workers": num_workers > 0}
    return {name: DataLoader(dataset, **kwargs) for name, dataset in datasets.items()}, indices


def save_split_trajectories(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    indices: dict[str, Tensor],
    output_dir: str | Path,
    *,
    device: torch.device,
    forward_args: Any,
    metadata: dict[str, Any],
    expected_test_accuracy: float | None = None,
    include_hidden_features: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"splits": {}}
    payloads: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        payload = export_trajectory_payload(model, loaders[split], indices[split], split=split,
                                            device=device, forward_args=forward_args, metadata=metadata,
                                            include_hidden_features=include_hidden_features)
        payloads[split] = payload
        accuracy = payload["correct"][:, -1].float().mean().item() * 100.0
        summary["splits"][split] = {"samples": len(payload["targets"]), "final_accuracy": accuracy}
    if expected_test_accuracy is not None and abs(summary["splits"]["test"]["final_accuracy"] - expected_test_accuracy) > 1e-4:
        raise RuntimeError("Exported test trajectory accuracy does not match summary.json.")
    for split, payload in payloads.items():
        torch.save(payload, output_dir / f"{split}_trajectories.pt")
    with (output_dir / "trajectory_export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
