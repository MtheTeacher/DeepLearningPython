"""Modernized training utilities for the "Neural Networks and Deep Learning" course."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort for editable installs
    __version__ = version("deeplearning-python")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "run_notebook_training"]


def __getattr__(name: str):  # pragma: no cover - tiny import shim
    if name == "run_notebook_training":
        from .notebook import run_notebook_training as helper

        return helper
    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(__all__)
