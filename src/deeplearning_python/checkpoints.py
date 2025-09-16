"""Checkpoint utilities for saving and resuming experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class CheckpointState:
    step: int
    epoch: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]


def save_checkpoint(
    *,
    path: Path,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    return path


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> CheckpointState:
    data = torch.load(Path(path), map_location=map_location)
    return CheckpointState(
        step=int(data["step"]),
        epoch=int(data["epoch"]),
        model_state=data["model_state"],
        optimizer_state=data["optimizer_state"],
        scheduler_state=data.get("scheduler_state"),
    )
