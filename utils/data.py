"""Dataset and DataLoader helpers."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "fashionmnist": {
        "num_classes": 10,
        "input_channels": 1,
        "input_size": 28,
        "is_temporal": False,
    },
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "input_size": 32,
        "is_temporal": False,
    },
    "nmnist": {
        "num_classes": 10,
        "input_channels": 2,
        "input_size": 34,
        "is_temporal": True,
    },
    "dvs_gesture": {
        "num_classes": 11,
        "input_channels": 2,
        "input_size": 64,
        "is_temporal": True,
    },
}

_DATASET_ALIASES = {
    "fashion-mnist": "fashionmnist",
    "mnist": "fashionmnist",
    "cifar-10": "cifar10",
    "dvs-gesture": "dvs_gesture",
    "dvsgesture": "dvs_gesture",
}


def canonical_dataset_name(dataset: str) -> str:
    name = dataset.lower()
    return _DATASET_ALIASES.get(name, name)


def get_dataset_spec(dataset: str) -> dict[str, Any]:
    name = canonical_dataset_name(dataset)
    if name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_SPECS[name].copy()


def is_temporal_dataset(dataset: str) -> bool:
    return bool(get_dataset_spec(dataset)["is_temporal"])


def _require_tonic(dataset: str):
    try:
        import tonic
    except ImportError as exc:
        raise ImportError(f"Event dataset '{dataset}' requires tonic. Please install requirements.txt.") from exc
    return tonic


class EventFrameTransform:
    def __init__(
        self,
        *,
        sensor_size: tuple[int, int, int],
        tmax: int,
        frame_mode: str,
        downsample_size: int | None,
        expected_channels: int,
    ) -> None:
        if frame_mode not in {"binary", "count"}:
            raise ValueError(f"Unsupported event_frame_mode: {frame_mode}")
        tonic = _require_tonic("event")
        self.to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=tmax)
        self.tmax = tmax
        self.frame_mode = frame_mode
        self.downsample_size = downsample_size
        self.expected_channels = expected_channels

    def __call__(self, events: object) -> torch.Tensor:
        frames = torch.as_tensor(self.to_frame(events), dtype=torch.float32)
        if frames.ndim != 4:
            raise ValueError(f"Expected event frames [T,C,H,W], got {tuple(frames.shape)}")
        if frames.shape[0] != self.tmax:
            raise ValueError(f"Expected {self.tmax} time bins, got {frames.shape[0]}")
        if frames.shape[1] != self.expected_channels:
            raise ValueError(
                f"Expected {self.expected_channels} polarity channels at axis 1, got shape {tuple(frames.shape)}"
            )
        if self.frame_mode == "binary":
            frames = (frames > 0).to(torch.float32)
        if self.downsample_size is not None and frames.shape[-2:] != (self.downsample_size, self.downsample_size):
            output_size = (self.downsample_size, self.downsample_size)
            if self.frame_mode == "binary":
                frames = F.adaptive_max_pool2d(frames, output_size)
                frames = (frames > 0).to(torch.float32)
            else:
                old_h, old_w = frames.shape[-2:]
                new_h, new_w = output_size
                frames = F.adaptive_avg_pool2d(frames, output_size)
                frames = frames * (float(old_h * old_w) / float(new_h * new_w))
        return frames


def _build_event_datasets(
    dataset: str,
    data_dir: str,
    *,
    tmax: int,
    event_frame_mode: str,
    event_downsample_size: int | None,
):
    tonic = _require_tonic(dataset)
    name = canonical_dataset_name(dataset)
    spec = get_dataset_spec(name)
    expected_channels = int(spec["input_channels"])
    if name == "nmnist":
        dataset_cls = tonic.datasets.NMNIST
        sensor_size = dataset_cls.sensor_size
        downsample_size = event_downsample_size
    elif name == "dvs_gesture":
        dataset_cls = tonic.datasets.DVSGesture
        sensor_size = dataset_cls.sensor_size
        downsample_size = 64 if event_downsample_size is None else event_downsample_size
    else:
        raise ValueError(f"Unsupported event dataset: {dataset}")

    transform = EventFrameTransform(
        sensor_size=sensor_size,
        tmax=tmax,
        frame_mode=event_frame_mode,
        downsample_size=downsample_size,
        expected_channels=expected_channels,
    )
    train_set = dataset_cls(save_to=data_dir, train=True, transform=transform)
    test_set = dataset_cls(save_to=data_dir, train=False, transform=transform)
    return train_set, test_set


def build_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    *,
    tmax: int,
    num_workers: int | None = None,
    event_frame_mode: str = "binary",
    event_downsample_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    dataset = canonical_dataset_name(dataset)
    if num_workers is None:
        num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))

    if dataset == "fashionmnist":
        train_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        test_tf = train_tf
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_tf)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)
    elif dataset == "cifar10":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        test_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    elif is_temporal_dataset(dataset):
        train_set, test_set = _build_event_datasets(
            dataset,
            data_dir,
            tmax=tmax,
            event_frame_mode=event_frame_mode,
            event_downsample_size=event_downsample_size,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
