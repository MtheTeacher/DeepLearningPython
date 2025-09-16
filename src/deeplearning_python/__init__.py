"""Modernized training utilities for the "Neural Networks and Deep Learning" course."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - best effort for editable installs
    __version__ = version("deeplearning-python")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
