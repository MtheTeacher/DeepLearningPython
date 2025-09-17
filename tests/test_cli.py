from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")

from deeplearning_python.cli import app
from deeplearning_python.models import SIMPLE_MLP_PRESETS


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


def test_cli_disables_live_for_non_tty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deeplearning_python import cli as cli_module

    messages: list[str] = []

    class DummyConsole:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.is_terminal = False

        def log(self, message: str) -> None:
            messages.append(message)

        def print(self, *_: Any, **__: Any) -> None:
            pass

    monkeypatch.setattr(cli_module, "_console", lambda verbose: DummyConsole())

    captured: dict[str, Any] = {}

    class DummyTrainer:
        def __init__(self, *, training_config, console, **kwargs: Any) -> None:
            captured["enable_live"] = training_config.enable_live
            captured["console"] = console

        def train(self) -> None:
            pass

    monkeypatch.setattr(cli_module, "Trainer", DummyTrainer)

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
    assert captured["enable_live"] is False
    assert messages, "expected a notice about disabling the live dashboard"


def test_cli_applies_model_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from deeplearning_python import cli as cli_module

    captured: dict[str, Any] = {}

    class DummyTrainer:
        def __init__(self, *, model_config, **kwargs: Any) -> None:
            captured["hidden_sizes"] = model_config.hidden_sizes
            captured["dropout"] = model_config.dropout

        def train(self) -> None:
            pass

    monkeypatch.setattr(cli_module, "Trainer", DummyTrainer)

    preset_name = "wide_dropout"
    result = runner.invoke(
        app,
        [
            "train",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.01",
            "--optimizer",
            "sgd",
            "--device",
            "cpu",
            "--fake-data",
            "--no-live",
            "--model-preset",
            preset_name,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.stdout
    expected_hidden, expected_dropout = SIMPLE_MLP_PRESETS[preset_name]
    assert captured["hidden_sizes"] == expected_hidden
    assert captured["dropout"] == expected_dropout


def test_cli_hidden_sizes_override_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    from deeplearning_python import cli as cli_module

    captured: dict[str, Any] = {}

    class DummyTrainer:
        def __init__(self, *, model_config, **kwargs: Any) -> None:
            captured["hidden_sizes"] = model_config.hidden_sizes
            captured["dropout"] = model_config.dropout

        def train(self) -> None:
            pass

    monkeypatch.setattr(cli_module, "Trainer", DummyTrainer)

    result = runner.invoke(
        app,
        [
            "train",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.01",
            "--optimizer",
            "sgd",
            "--device",
            "cpu",
            "--fake-data",
            "--no-live",
            "--model-preset",
            "wide_dropout",
            "--hidden-sizes",
            "32",
            "--hidden-sizes",
            "16",
            "--dropout",
            "0.2",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.stdout
    assert captured["hidden_sizes"] == (32, 16)
    assert captured["dropout"] == pytest.approx(0.2)
