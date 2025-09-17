from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from deeplearning_python.notebook import run_notebook_training


def test_notebook_helper_returns_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from deeplearning_python import training as training_module

    live_called = {"value": False}

    class SentinelLive:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - not executed
            live_called["value"] = True

        def start(self) -> None:  # pragma: no cover - not executed
            pass

        def update(self, *args, **kwargs) -> None:  # pragma: no cover - not executed
            pass

        def stop(self) -> None:  # pragma: no cover - not executed
            pass

    monkeypatch.setattr(training_module, "Live", SentinelLive)

    result = run_notebook_training(
        epochs=1,
        batch_size=8,
        val_batch_size=8,
        learning_rate=0.01,
        optimizer="sgd",
        device="cpu",
        fake_data=True,
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpts",
        metrics_filename="metrics.jsonl",
        limit_train_batches=1,
        limit_val_batches=1,
        limit_train_samples=16,
        limit_val_samples=16,
        preview_interval=0,
        log_interval=1,
        checkpoint_interval=0,
        verbose=False,
    )

    assert not live_called["value"], "live dashboard should stay disabled in notebooks"

    metrics = result["metrics"]
    assert len(metrics) > 0, "expected at least one training step"

    log_path = Path(result["log_path"])
    assert log_path.exists()
    assert log_path.read_text().strip(), "metrics log should contain entries"
