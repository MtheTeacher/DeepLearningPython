"""Helpers for running the trainer from interactive notebooks."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, MutableMapping

from rich.console import Console

from .data import DataConfig
from .models import ModelConfig
from .training import Trainer, TrainingConfig

__all__ = ["run_notebook_training"]


def run_notebook_training(
    *,
    model: str = "simple",
    model_preset: str | None = "baseline",
    hidden_sizes: Iterable[int] | None = None,
    dropout: float | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    val_batch_size: int | None = None,
    learning_rate: float = 0.1,
    optimizer: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    scheduler: str = "none",
    scheduler_step_size: int = 1,
    scheduler_gamma: float = 0.95,
    device: str = "auto",
    data_dir: str | Path = "./data",
    download: bool = True,
    fake_data: bool = False,
    checkpoint_dir: str | Path = "./artifacts/checkpoints",
    checkpoint_interval: int = 0,
    log_dir: str | Path = "./artifacts/logs",
    metrics_filename: str = "metrics.jsonl",
    resume_from: str | Path | None = None,
    log_interval: int = 10,
    preview_interval: int = 25,
    evaluate_every: int = 1,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
    mixed_precision: bool = False,
    gradient_clip: float | None = None,
    num_workers: int = 0,
    seed: int = 0,
    verbose: bool = True,
    return_dataframe: bool = False,
) -> MutableMapping[str, Any]:
    """Train the classroom model from a notebook and return the collected metrics.

    Parameters mirror the Typer CLI defaults while forcing ``enable_live`` off so
    notebook output remains readable. ``model_preset`` accepts the same names as
    ``dlp train`` and can be overridden with ``hidden_sizes`` or ``dropout`` when
    you want finer control. Set ``return_dataframe`` when you prefer a pandas
    ``DataFrame`` instead of a list of ``StepMetrics`` objects.
    """

    resolved_hidden_sizes = tuple(hidden_sizes) if hidden_sizes is not None else None

    preset = (
        model_preset.lower() if (model == "simple" and model_preset is not None) else None
    )
    model_config = ModelConfig(
        name=model,
        preset=preset,
        hidden_sizes=resolved_hidden_sizes,
        dropout=dropout,
    )
    data_config = DataConfig(
        data_dir=Path(data_dir),
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        download=download,
        num_workers=num_workers,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
        seed=seed,
        use_fake_data=fake_data,
    )
    training_config = TrainingConfig(
        epochs=epochs,
        optimizer=optimizer,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        device=device,
        checkpoint_dir=Path(checkpoint_dir),
        checkpoint_interval=checkpoint_interval,
        log_dir=Path(log_dir),
        metrics_filename=metrics_filename,
        resume_from=Path(resume_from) if resume_from else None,
        preview_interval=preview_interval,
        log_interval=log_interval,
        evaluate_every=evaluate_every,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
        mixed_precision=mixed_precision,
        gradient_clip=gradient_clip,
        scheduler=scheduler,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        num_workers=num_workers,
        seed=seed,
        enable_live=False,
    )

    console = Console(log_path=False, quiet=not verbose)
    trainer = Trainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        console=console,
    )
    trainer.train()

    history = list(trainer.metrics_tracker.history)
    metrics: Any = history
    if return_dataframe:
        try:
            import pandas as pd  # type: ignore
        except ImportError as error:  # pragma: no cover - exercised when pandas missing
            raise RuntimeError(
                "pandas is required when return_dataframe=True; install pandas first."
            ) from error
        metrics = pd.DataFrame([asdict(entry) for entry in history])

    return {
        "metrics": metrics,
        "log_path": trainer.log_path,
        "checkpoint_dir": Path(training_config.checkpoint_dir),
    }
