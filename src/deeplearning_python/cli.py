"""Command line interface for the classroom trainer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
import click
from rich.console import Console
from click.core import ParameterSource
from typer.main import TyperChoice

from .data import DataConfig
from .models import ModelConfig, SIMPLE_MLP_PRESETS


SIMPLE_MLP_PRESET_CHOICE = TyperChoice(
    list(SIMPLE_MLP_PRESETS.keys()), case_sensitive=False
)
from .training import Trainer, TrainingConfig

app = typer.Typer(help="Interactive training utilities for the classroom demos.")


def _console(verbose: bool) -> Console:
    return Console(log_path=False, quiet=not verbose)


@app.command()
def train(
    model: str = typer.Option(
        "simple",
        help="Which model architecture to use (simple, regularized, conv).",
    ),
    model_preset: str = typer.Option(
        "baseline",
        "--model-preset",
        help="Preset hidden sizes and dropout for the simple MLP (only used with --model simple).",
        case_sensitive=False,
        click_type=SIMPLE_MLP_PRESET_CHOICE,
    ),
    hidden_sizes: List[int] = typer.Option(
        [128, 64],
        help="Hidden layer sizes for the simple MLP (only used with --model simple). Overrides --model-preset when provided.",
        show_default=False,
    ),
    dropout: Optional[float] = typer.Option(
        None,
        help="Dropout probability for the simple MLP (only used with --model simple; falls back to the preset value).",
    ),
    epochs: int = typer.Option(5, help="Number of training epochs."),
    batch_size: int = typer.Option(64, help="Mini-batch size for SGD."),
    val_batch_size: Optional[int] = typer.Option(
        None, help="Validation batch size (defaults to batch-size)."
    ),
    learning_rate: float = typer.Option(0.1, help="Initial learning rate."),
    optimizer: str = typer.Option(
        "sgd", help="Optimizer to use (sgd or adam)."
    ),
    momentum: float = typer.Option(0.9, help="Momentum for SGD."),
    weight_decay: float = typer.Option(0.0, help="L2 regularization weight."),
    scheduler: str = typer.Option(
        "none",
        help="Learning rate scheduler (none, steplr, cosine).",
    ),
    scheduler_step_size: int = typer.Option(
        1, help="StepLR step size (epochs)."
    ),
    scheduler_gamma: float = typer.Option(
        0.95, help="StepLR gamma multiplier."
    ),
    device: str = typer.Option(
        "auto", help="Device to run on (auto, cpu, cuda, cuda:0, ...)."
    ),
    data_dir: Path = typer.Option(
        Path("./data"), help="Directory to download/cache MNIST data."
    ),
    download: bool = typer.Option(True, help="Download MNIST if missing."),
    fake_data: bool = typer.Option(
        False, help="Use synthetic data instead of downloading MNIST."
    ),
    checkpoint_dir: Path = typer.Option(
        Path("./artifacts/checkpoints"), help="Directory to store checkpoints."
    ),
    checkpoint_interval: int = typer.Option(
        200, help="Number of optimizer steps between checkpoints (0 disables)."
    ),
    resume_from: Optional[Path] = typer.Option(
        None, help="Resume training from a saved checkpoint."
    ),
    log_dir: Path = typer.Option(
        Path("./artifacts/logs"), help="Directory to write metrics."
    ),
    metrics_filename: str = typer.Option(
        "metrics.jsonl", help="Name of the JSONL metrics file."
    ),
    log_interval: int = typer.Option(
        10, help="Number of steps between metrics log entries."
    ),
    preview_interval: int = typer.Option(
        25, help="Number of steps between mini-batch previews."
    ),
    evaluate_every: int = typer.Option(
        1, help="How many epochs between validation runs."
    ),
    limit_train_batches: Optional[int] = typer.Option(
        None, help="Limit the number of training batches per epoch (demo mode)."
    ),
    limit_val_batches: Optional[int] = typer.Option(
        None, help="Limit the number of validation batches per epoch."
    ),
    limit_train_samples: Optional[int] = typer.Option(
        None, help="Limit the number of training samples (subsampling)."
    ),
    limit_val_samples: Optional[int] = typer.Option(
        None, help="Limit the number of validation samples (subsampling)."
    ),
    mixed_precision: bool = typer.Option(
        False, help="Enable mixed-precision training when CUDA is available."
    ),
    gradient_clip: Optional[float] = typer.Option(
        None, help="Clip gradients at this global norm if provided."
    ),
    num_workers: int = typer.Option(
        0, help="Number of worker processes for the data loader."
    ),
    seed: int = typer.Option(0, help="Random seed."),
    verbose: bool = typer.Option(True, help="Print progress to the console."),
    live: bool = typer.Option(  # noqa: B008
        True,
        "--live/--no-live",
        help="Enable the live Rich dashboard (disable for tests or logging-only runs).",
    ),
) -> None:
    """Train a network with a live terminal dashboard."""

    console = _console(verbose)

    live_mode = live
    if live_mode and not console.is_terminal:
        console.log(
            "Detected a non-interactive output stream; disabling the live dashboard."
        )
        live_mode = False
    preset = model_preset.lower() if model == "simple" else None
    ctx = click.get_current_context()
    hidden_sizes_source = ctx.get_parameter_source("hidden_sizes")
    hidden_sizes_override = hidden_sizes_source not in (
        ParameterSource.DEFAULT,
        ParameterSource.DEFAULT_MAP,
    )
    resolved_hidden_sizes = tuple(hidden_sizes) if hidden_sizes_override else None
    model_config = ModelConfig(
        name=model,
        preset=preset,
        hidden_sizes=resolved_hidden_sizes,
        dropout=dropout,
    )

    data_config = DataConfig(
        data_dir=data_dir,
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
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        log_dir=log_dir,
        metrics_filename=metrics_filename,
        resume_from=resume_from,
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
        enable_live=live_mode,
    )

    trainer = Trainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        console=console,
    )
    trainer.train()


@app.command()
def demo(
    steps: int = typer.Option(
        200, help="Number of SGD steps to run for the classroom demo."
    ),
    **options,
) -> None:
    """Run a short interactive demo with aggressive logging."""

    options.setdefault("epochs", 1)
    options.setdefault("limit_train_batches", steps)
    options.setdefault("evaluate_every", 1)
    options.setdefault("preview_interval", 5)
    options.setdefault("log_interval", 5)
    options.setdefault("verbose", True)
    train(**options)


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
