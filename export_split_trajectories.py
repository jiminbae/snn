#!/usr/bin/env python3
"""Export deterministic train/validation/test logits from a selected backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models import build_model
from utils.data import _build_datasets, get_dataset_spec
from utils.trajectory_export import build_export_loaders, load_torch_compat, save_split_trajectories

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-hidden-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    required = ["config.json", "best_checkpoint.pt", "split_indices.pt", "selection_summary.json", "summary.json"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run directory is missing required files: {missing}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("limit_test_batches"):
        raise ValueError(
            "Cannot export a full official test trajectory from a run whose summary used --limit-test-batches."
        )
    selection = json.loads((run_dir / "selection_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    split_indices = load_torch_compat(run_dir / "split_indices.pt")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    downsample = config.get("resolved_event_downsample_size", config.get("event_downsample_size"))
    train_dataset, test_dataset = _build_datasets(
        config["dataset"], config.get("data_dir", "data"), tmax=config["tmax"],
        event_frame_mode=config.get("event_frame_mode", "binary"), event_downsample_size=downsample,
    )
    loaders, indices = build_export_loaders(train_dataset, test_dataset, split_indices, args.batch_size, args.num_workers)
    model = build_model(config["model"], dataset=config["dataset"], tmax=config["tmax"],
                        gate_init=config.get("gate_init", 5.0)).to(device)
    checkpoint = load_torch_compat(run_dir / "best_checkpoint.pt", device)
    model.load_state_dict(checkpoint["model_state_dict"])
    forward_args = SimpleNamespace(
        gate_threshold=config.get("gate_threshold", 0.5), min_prefix_steps=config.get("min_prefix_steps", 1),
        temporal_prefix_steps=config.get("temporal_prefix_steps", 0),
        temporal_prefix_mode=config.get("temporal_prefix_mode", "none"),
    )
    c1 = int(model.backbone.conv1.out_channels)
    c2 = int(model.backbone.conv2.out_channels)
    hidden_feature_metadata = {
        "format_version": 1,
        "representation": "channelwise_spatial_mean",
        "uses_target": False,
        "causal": True,
        "dimension": 2 * c1 + 2 * c2,
        "groups": [
            {"name": "u1_mean", "start": 0, "stop": c1},
            {"name": "s1_mean", "start": c1, "stop": 2 * c1},
            {"name": "u2_mean", "start": 2 * c1, "stop": 2 * c1 + c2},
            {"name": "s2_mean", "start": 2 * c1 + c2, "stop": 2 * c1 + 2 * c2},
        ],
    }
    fingerprints = {name: sha256_file(run_dir / name) for name in required}
    metadata = {"dataset": config["dataset"], "model": config["model"], "tmax": config["tmax"],
                "num_classes": int(get_dataset_spec(config["dataset"])["num_classes"]),
                "checkpoint_epoch": checkpoint["epoch"], "split_seed": split_indices["split_seed"],
                "backbone_seed": config.get("seed", 0), "method": run_dir.parent.name,
                "source_fingerprints": fingerprints,
                "hidden_feature_metadata": hidden_feature_metadata,
                "hidden_features_included": args.include_hidden_features}
    result = save_split_trajectories(model, loaders, indices, run_dir / "trajectories", device=device,
                                     forward_args=forward_args, metadata=metadata,
                                     expected_test_accuracy=summary["test_accuracy"],
                                     include_hidden_features=args.include_hidden_features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
