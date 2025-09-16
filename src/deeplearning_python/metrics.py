"""Utility classes for accumulating training metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List

import torch


@dataclass
class StepMetrics:
    step: int
    epoch: int
    loss: float
    accuracy: float
    gradient_norm: float
    learning_rate: float


@dataclass
class MetricsTracker:
    history: List[StepMetrics] = field(default_factory=list)

    def update(
        self,
        *,
        step: int,
        epoch: int,
        loss: torch.Tensor,
        accuracy: float,
        gradient_norm: float,
        learning_rate: float,
    ) -> StepMetrics:
        metric = StepMetrics(
            step=step,
            epoch=epoch,
            loss=float(loss.detach().cpu().item()),
            accuracy=float(accuracy),
            gradient_norm=float(gradient_norm),
            learning_rate=float(learning_rate),
        )
        self.history.append(metric)
        return metric

    def latest(self) -> StepMetrics | None:
        return self.history[-1] if self.history else None

    def summary(self) -> Dict[str, float]:
        if not self.history:
            return {}
        return {
            "loss": mean(metric.loss for metric in self.history),
            "accuracy": mean(metric.accuracy for metric in self.history),
            "gradient_norm": mean(metric.gradient_norm for metric in self.history),
            "learning_rate": mean(metric.learning_rate for metric in self.history),
        }
