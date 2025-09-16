"""Model definitions inspired by the book's networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Type

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """A small MLP similar to the book's original network."""

    def __init__(self, hidden_sizes: Tuple[int, ...] = (128, 64), dropout: float = 0.0):
        super().__init__()
        layers = []
        input_size = 28 * 28
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple forwarding
        x = x.view(x.size(0), -1)
        return self.network(x)


class RegularizedMLP(nn.Module):
    """MLP with batch norm and dropout to mimic the book's improved network."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = x.view(x.size(0), -1)
        return self.network(x)


class ConvNet(nn.Module):
    """A small convolutional network for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.features(x)
        return self.classifier(x)


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simple": SimpleMLP,
    "regularized": RegularizedMLP,
    "conv": ConvNet,
}


@dataclass
class ModelConfig:
    name: str = "simple"
    hidden_sizes: Tuple[int, ...] = (128, 64)
    dropout: float = 0.0

    def create_model(self) -> nn.Module:
        if self.name == "simple":
            return SimpleMLP(hidden_sizes=self.hidden_sizes, dropout=self.dropout)
        if self.name == "regularized":
            return RegularizedMLP()
        if self.name == "conv":
            return ConvNet()
        raise ValueError(f"Unknown model name: {self.name}")
