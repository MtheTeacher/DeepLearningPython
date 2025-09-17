"""Model definitions inspired by the book's networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

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


SIMPLE_MLP_PRESETS: Dict[str, Tuple[Tuple[int, ...], float]] = {
    "baseline": ((128, 64), 0.0),
    "compact": ((64, 32), 0.0),
    "wide_dropout": ((256, 128, 64), 0.1),
}


@dataclass
class ModelConfig:
    name: str = "simple"
    hidden_sizes: Optional[Tuple[int, ...]] = None
    dropout: Optional[float] = None
    preset: Optional[str] = None

    def __post_init__(self) -> None:
        if self.name == "simple":
            preset_hidden: Optional[Tuple[int, ...]]
            preset_dropout: Optional[float]
            if self.preset is not None:
                preset_name = self.preset
            elif self.hidden_sizes is None:
                preset_name = "baseline"
            else:
                preset_name = None

            if preset_name is None:
                preset_hidden = None
                preset_dropout = None
            else:
                try:
                    preset_hidden, preset_dropout = SIMPLE_MLP_PRESETS[preset_name]
                except KeyError as error:
                    available = ", ".join(sorted(SIMPLE_MLP_PRESETS))
                    raise ValueError(
                        f"Unknown simple MLP preset: {preset_name!r}. "
                        f"Available presets: {available}."
                    ) from error
                self.preset = preset_name

            if self.hidden_sizes is None:
                if preset_hidden is None:
                    raise ValueError(
                        "hidden_sizes must be provided when preset is None for the simple model."
                    )
                self.hidden_sizes = preset_hidden
            else:
                self.hidden_sizes = tuple(self.hidden_sizes)

            if self.dropout is None:
                self.dropout = preset_dropout if preset_dropout is not None else 0.0
            else:
                # When both a preset and explicit dropout are provided, the explicit
                # value wins while still allowing presets to document their defaults.
                self.dropout = float(self.dropout)
        else:
            if self.preset is not None:
                raise ValueError(
                    "Model presets are only supported for the 'simple' architecture."
                )
            if self.hidden_sizes is not None:
                raise ValueError(
                    "hidden_sizes is only configurable when name='simple'."
                )
            if self.dropout not in (None, 0.0):
                raise ValueError(
                    "dropout is only configurable for the 'simple' architecture."
                )
            self.hidden_sizes = tuple()
            self.dropout = 0.0

        assert self.hidden_sizes is not None
        assert self.dropout is not None

    def create_model(self) -> nn.Module:
        if self.name == "simple":
            return SimpleMLP(hidden_sizes=self.hidden_sizes, dropout=self.dropout)
        if self.name == "regularized":
            return RegularizedMLP()
        if self.name == "conv":
            return ConvNet()
        raise ValueError(f"Unknown model name: {self.name}")
