"""Dataset utilities for the classroom training demos."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    """Configuration for loading MNIST datasets."""

    data_dir: Path
    batch_size: int = 64
    val_batch_size: Optional[int] = None
    download: bool = True
    num_workers: int = 0
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    seed: int = 0
    use_fake_data: bool = False

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.val_batch_size = self.val_batch_size or self.batch_size
        self.data_dir.mkdir(parents=True, exist_ok=True)


def _limit_dataset(dataset: Dataset, limit: Optional[int], seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    limited = indices[: limit]
    return Subset(dataset, limited)


def mnist_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_mnist_datasets(config: DataConfig) -> Tuple[Dataset, Dataset]:
    transform = mnist_transforms()
    if config.use_fake_data:
        train_size = config.limit_train_samples or 512
        val_size = config.limit_val_samples or 128
        train_dataset = datasets.FakeData(
            size=train_size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=config.seed,
        )
        val_dataset = datasets.FakeData(
            size=val_size,
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transform,
            random_offset=config.seed + 1,
        )
    else:
        train_dataset = datasets.MNIST(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=transform,
        )
        val_dataset = datasets.MNIST(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=transform,
        )
        train_dataset = _limit_dataset(train_dataset, config.limit_train_samples, config.seed)
        val_dataset = _limit_dataset(val_dataset, config.limit_val_samples, config.seed)
    return train_dataset, val_dataset


def get_mnist_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = get_mnist_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size or config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def iter_batch_preview(
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    max_batches: int = 1,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Grab a few batches for visualization without exhausting the loader."""

    previews: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _, (images, labels) in zip(range(max_batches), batches):
        previews.append((images, labels))
    return previews
