from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")

from deeplearning_python.cli import app


runner = CliRunner()


def test_cli_train_fake_data(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_dir = tmp_path / "ckpts"
    result = runner.invoke(
        app,
        [
            "train",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.01",
            "--optimizer",
            "sgd",
            "--device",
            "cpu",
            "--fake-data",
            "--no-live",
            "--log-dir",
            str(tmp_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--metrics-filename",
            "metrics.jsonl",
            "--limit-train-batches",
            "1",
            "--limit-val-batches",
            "1",
            "--limit-train-samples",
            "16",
            "--limit-val-samples",
            "16",
            "--checkpoint-interval",
            "0",
            "--log-interval",
            "1",
            "--preview-interval",
            "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    assert metrics_path.exists()
    contents = metrics_path.read_text().strip().splitlines()
    assert contents, "expected metrics log to contain at least one entry"
    assert any('"phase": "train"' in line for line in contents)
    assert not list(checkpoint_dir.glob("*.pt")), "checkpoints should be disabled"
