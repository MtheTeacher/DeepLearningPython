"""Training loop and classroom demo orchestration."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from rich.console import Console
from rich.live import Live
from torch import nn
from torch.optim import Optimizer

from .checkpoints import load_checkpoint, save_checkpoint
from .data import DataConfig, get_mnist_dataloaders, iter_batch_preview
from .metrics import MetricsTracker
from .models import ModelConfig
from .visualization import LivePreview, TrainingDashboard


@dataclass
class TrainingConfig:
    epochs: int = 5
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 64
    val_batch_size: Optional[int] = None
    device: str = "auto"
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    checkpoint_interval: int = 200
    log_dir: Path = Path("artifacts/logs")
    metrics_filename: str = "metrics.jsonl"
    resume_from: Optional[Path] = None
    preview_interval: int = 25
    log_interval: int = 10
    evaluate_every: int = 1
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    mixed_precision: bool = False
    gradient_clip: Optional[float] = None
    scheduler: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.95
    num_workers: int = 0
    seed: int = 0
    enable_live: bool = True

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) / self.metrics_filename


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(
    optimizer: Optimizer, config: TrainingConfig
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.scheduler == "none":
        return None
    if config.scheduler == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs, 1),
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = predictions.argmax(dim=1)
    return float((predicted == labels).float().mean().item())


def compute_gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().data.norm(2).item() ** 2
    return math.sqrt(total)


class Trainer:
    """High-level orchestration for the interactive classroom trainer."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        console: Optional[Console] = None,
    ) -> None:
        self.console = console or Console()
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.device = self.training_config.resolve_device()
        self.dashboard = TrainingDashboard(console=self.console)
        self.metrics_tracker = MetricsTracker()
        self.global_step = 0

        self.log_path = self.training_config.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w", encoding="utf8")

    # region lifecycle helpers
    def _load_data(self):
        data_config = DataConfig(
            data_dir=self.data_config.data_dir,
            batch_size=self.training_config.batch_size,
            val_batch_size=self.training_config.val_batch_size,
            download=self.data_config.download,
            num_workers=self.training_config.num_workers,
            limit_train_samples=self.training_config.limit_train_samples,
            limit_val_samples=self.training_config.limit_val_samples,
            seed=self.training_config.seed,
            use_fake_data=self.data_config.use_fake_data,
        )
        return get_mnist_dataloaders(data_config)

    def _create_model(self) -> nn.Module:
        model = self.model_config.create_model()
        model.to(self.device)
        return model

    def _write_metrics_log(self, payload: Dict[str, object]) -> None:
        json.dump(payload, self._log_file)
        self._log_file.write("\n")
        self._log_file.flush()

    def _restore_checkpoint(
        self, model: nn.Module, optimizer: Optimizer, scheduler
    ) -> Dict[str, int]:
        if not self.training_config.resume_from:
            return {"step": 0, "epoch": 0}
        state = load_checkpoint(self.training_config.resume_from, map_location=self.device)
        model.load_state_dict(state.model_state)
        optimizer.load_state_dict(state.optimizer_state)
        if scheduler and state.scheduler_state:
            scheduler.load_state_dict(state.scheduler_state)
        self.console.print(f"Resumed from {self.training_config.resume_from} at step {state.step}")
        return {"step": state.step, "epoch": state.epoch}

    def _maybe_checkpoint(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
        epoch: int,
        step: int,
    ) -> None:
        interval = self.training_config.checkpoint_interval
        if interval <= 0 or step % interval != 0:
            return
        checkpoint_dir = Path(self.training_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"step_{step:07d}.pt"
        save_checkpoint(
            path=path,
            step=step,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata={"timestamp": time.time()},
        )
        self.console.print(f"Saved checkpoint to {path}")

    # endregion

    def train(self) -> None:
        torch.manual_seed(self.training_config.seed)
        train_loader, val_loader = self._load_data()
        model = self._create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, self.training_config)
        scheduler = build_scheduler(optimizer, self.training_config)

        scaler = torch.cuda.amp.GradScaler() if self.training_config.mixed_precision else None
        state = self._restore_checkpoint(model, optimizer, scheduler)
        self.global_step = state.get("step", 0)

        initial_preview = iter_batch_preview(train_loader, max_batches=1)
        if initial_preview:
            images, labels = initial_preview[0]
            images = images.detach().cpu()
            labels = labels.detach().cpu()
            self.dashboard.update_preview(
                LivePreview(images=images, labels=labels)
            )

        self.console.print(
            f"Starting training for {self.training_config.epochs} epochs on {self.device}"
        )

        live: Live | None = None
        if self.training_config.enable_live:
            live = Live(self.dashboard.render(), console=self.console, refresh_per_second=4)
            live.start()
        try:
            for epoch in range(state.get("epoch", 0), self.training_config.epochs):
                model.train()
                for batch_index, (images, labels) in enumerate(train_loader, start=1):
                    if (
                        self.training_config.limit_train_batches
                        and batch_index > self.training_config.limit_train_batches
                    ):
                        break
                    self.global_step += 1
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=bool(scaler)):
                        logits = model(images)
                        loss = criterion(logits, labels)
                    preds = logits.detach()

                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    gradient_norm = compute_gradient_norm(model)
                    if self.training_config.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.training_config.gradient_clip
                        )

                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler:
                        scheduler.step()

                    accuracy = compute_accuracy(preds, labels)
                    learning_rate = optimizer.param_groups[0]["lr"]
                    metrics = self.metrics_tracker.update(
                        step=self.global_step,
                        epoch=epoch + 1,
                        loss=loss,
                        accuracy=accuracy,
                        gradient_norm=gradient_norm,
                        learning_rate=learning_rate,
                    )
                    self.dashboard.update_metrics(metrics)
                    if (
                        self.training_config.preview_interval
                        and self.training_config.preview_interval > 0
                        and self.global_step % self.training_config.preview_interval == 0
                    ):
                        preview_batches = iter_batch_preview([
                            (images.detach().cpu(), labels.detach().cpu())
                        ],
                            max_batches=1,
                        )
                        preview_images, preview_labels = preview_batches[0]
                        predicted_classes = preds.argmax(dim=1).cpu()
                        self.dashboard.update_preview(
                            LivePreview(
                                images=preview_images,
                                labels=preview_labels,
                                predictions=predicted_classes,
                            )
                        )

                    if (
                        self.training_config.log_interval
                        and self.training_config.log_interval > 0
                        and self.global_step % self.training_config.log_interval == 0
                    ):
                        self._write_metrics_log(
                            {
                                "phase": "train",
                                "step": self.global_step,
                                "epoch": epoch + 1,
                                "loss": metrics.loss,
                                "accuracy": metrics.accuracy,
                                "gradient_norm": metrics.gradient_norm,
                                "learning_rate": metrics.learning_rate,
                            }
                        )

                    self._maybe_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        step=self.global_step,
                    )

                    if live:
                        live.update(self.dashboard.render())

                if (epoch + 1) % max(self.training_config.evaluate_every, 1) == 0:
                    val_metrics = self.evaluate(model, val_loader, epoch + 1)
                    self._write_metrics_log({"phase": "val", **val_metrics})
                    self.console.print(
                        f"Epoch {epoch + 1}: val_loss={val_metrics['loss']:.4f} "
                        f"val_acc={val_metrics['accuracy']:.4f}"
                    )
                    if live:
                        live.update(self.dashboard.render())
        finally:
            if live:
                live.stop()
            self._log_file.close()

    def evaluate(
        self, model: nn.Module, val_loader: Iterable, epoch: int
    ) -> Dict[str, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        device = self.device
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(val_loader, start=1):
                if (
                    self.training_config.limit_val_batches
                    and batch_index > self.training_config.limit_val_batches
                ):
                    break
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                total_loss += float(loss.item()) * images.size(0)
                total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
                total_samples += int(images.size(0))
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        metrics = {
            "phase": "val",
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "step": self.global_step,
        }
        self.dashboard.log.append(
            f"Epoch {epoch} validation: loss {avg_loss:.4f}, accuracy {accuracy:.4f}"
        )
        return metrics
