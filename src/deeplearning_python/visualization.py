"""Rich-powered visualization for live training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Deque, Iterable, List, Optional

import torch
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.plot import Plot
from rich.progress import Progress
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .metrics import StepMetrics

ASCII_GRADIENT = " .:-=+*#%@"


def _chunk(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def tensor_to_ascii(image: torch.Tensor) -> str:
    image = image.detach().cpu()
    if image.ndim == 3:
        image = image.squeeze(0)
    image = image.float()
    # Try to undo MNIST normalization while remaining robust to other datasets
    image = image * 0.3081 + 0.1307
    image = image.clamp(0, 1)
    if torch.isclose(image.max(), image.min()):  # pragma: no cover - defensive guard
        image = torch.zeros_like(image)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rows = []
    for row in image:
        indices = (row * (len(ASCII_GRADIENT) - 1)).long()
        rows.append("".join(ASCII_GRADIENT[idx] for idx in indices))
    return "\n".join(rows)


@dataclass
class LivePreview:
    images: torch.Tensor
    labels: torch.Tensor
    predictions: Optional[torch.Tensor] = None


class TrainingDashboard:
    """Render live training metrics and previews in the terminal."""

    def __init__(
        self,
        console: Optional[Console] = None,
        history_size: int = 200,
        log_lines: int = 12,
        preview_digits: int = 6,
    ) -> None:
        self.console = console or Console()
        self.history: Deque[StepMetrics] = deque(maxlen=history_size)
        self.log: Deque[str] = deque(maxlen=log_lines)
        self.preview_digits = preview_digits
        self.preview: Optional[LivePreview] = None
        self.progress = Progress(transient=True)

    def update_metrics(self, metrics: StepMetrics) -> None:
        self.history.append(metrics)
        self.log.append(
            f"Step {metrics.step:,} | Epoch {metrics.epoch} | "
            f"Loss {metrics.loss:.4f} | Acc {metrics.accuracy:.4f} | "
            f"Grad {metrics.gradient_norm:.2f}"
        )

    def update_preview(self, preview: LivePreview) -> None:
        self.preview = preview

    # region Rendering helpers
    def _render_metrics_table(self) -> RenderableType:
        table = Table.grid(expand=True)
        table.add_column(justify="right", style="bold")
        table.add_column(justify="left")
        if self.history:
            latest = self.history[-1]
            table.add_row("Step", f"{latest.step:,}")
            table.add_row("Epoch", str(latest.epoch))
            table.add_row("Loss", f"{latest.loss:.4f}")
            table.add_row("Accuracy", f"{latest.accuracy:.4f}")
            table.add_row("Gradient Norm", f"{latest.gradient_norm:.2f}")
            table.add_row("Learning Rate", f"{latest.learning_rate:.6f}")
        else:
            table.add_row("Waiting", "Training not started")
        return Panel(table, title="Current Metrics", padding=(1, 2))

    def _render_loss_plot(self) -> RenderableType:
        plot = Plot(width=80, height=12, title="Loss History")
        if not self.history:
            plot.add_series("loss", [0.0, 0.0])
            return plot
        xs = [metric.step for metric in self.history]
        losses = [metric.loss for metric in self.history]
        plot.add_series("loss", losses, x_values=xs)
        return plot

    def _render_accuracy_plot(self) -> RenderableType:
        plot = Plot(width=80, height=12, title="Accuracy History")
        if not self.history:
            plot.add_series("accuracy", [0.0, 0.0])
            return plot
        xs = [metric.step for metric in self.history]
        accuracies = [metric.accuracy for metric in self.history]
        plot.add_series("accuracy", accuracies, x_values=xs)
        return plot

    def _render_preview_panel(self) -> RenderableType:
        if not self.preview:
            return Panel("Collecting a mini-batch preview...", title="Mini-batch View")

        images = self.preview.images[: self.preview_digits]
        labels = self.preview.labels[: self.preview_digits]
        predictions = (
            self.preview.predictions[: self.preview_digits]
            if self.preview.predictions is not None
            else None
        )

        ascii_digits = [tensor_to_ascii(img) for img in images]
        annotated_digits: List[str] = []
        for idx, art in enumerate(ascii_digits):
            label = int(labels[idx].item()) if labels is not None else "?"
            pred = (
                int(predictions[idx].item())
                if predictions is not None
                else "?"
            )
            title = f"Label {label}"
            if predictions is not None:
                title += f" | Pred {pred}"
            annotated_digits.append(f"{title}\n{art}")

        rows: List[str] = []
        columns = 3
        for chunk in _chunk(annotated_digits, columns):
            chunk_lines = [entry.splitlines() for entry in chunk]
            max_height = max(len(lines) for lines in chunk_lines)
            padded = [
                lines + [" " * len(lines[0])] * (max_height - len(lines))
                for lines in chunk_lines
            ]
            for row_idx in range(max_height):
                rows.append("   ".join(lines[row_idx] for lines in padded))
            rows.append("")
        preview_text = Text("\n".join(rows).rstrip())
        return Panel(preview_text, title="Mini-batch View", padding=(1, 2))

    def _render_log_panel(self) -> RenderableType:
        if not self.log:
            return Panel("Logs will appear here as training progresses.", title="Training Log")
        log_text = Text()
        for line in self.log:
            log_text.append(line + "\n")
        return Panel(log_text, title="Training Log", padding=(1, 2))

    def render(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(self._render_metrics_table(), size=10),
            Layout(Rule(style="dim"), size=1),
            Layout(self._render_loss_plot(), size=14),
            Layout(self._render_accuracy_plot(), size=14),
            Layout(self._render_preview_panel(), ratio=1),
            Layout(self._render_log_panel(), size=10),
        )
        return layout

    # endregion

    def refresh(self) -> None:
        """Convenience wrapper for manual refresh when not using Live."""
        self.console.print(self.render())
