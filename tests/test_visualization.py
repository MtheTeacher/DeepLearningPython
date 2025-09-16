from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deeplearning_python.visualization import tensor_to_ascii


def test_tensor_to_ascii_produces_expected_shape() -> None:
    image = torch.zeros((1, 28, 28))
    ascii_art = tensor_to_ascii(image)
    rows = ascii_art.splitlines()
    assert len(rows) == 28
    assert all(len(row) == 28 for row in rows)


def test_tensor_to_ascii_handles_constant_tensor() -> None:
    image = torch.ones((1, 28, 28)) * -0.5
    ascii_art = tensor_to_ascii(image)
    rows = ascii_art.splitlines()
    assert len(rows) == 28
    assert any(char.strip() == "" for char in rows[0])
