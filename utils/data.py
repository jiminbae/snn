"""Dataset and DataLoader helpers."""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    dataset = dataset.lower()
    if num_workers is None:
        num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))

    if dataset in {"fashionmnist", "fashion-mnist"}:
        train_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        test_tf = train_tf
        train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_tf)
        test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)
    elif dataset in {"cifar10", "cifar-10"}:
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
