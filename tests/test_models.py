from __future__ import annotations

import pytest

pytest.importorskip("torch")

from torch import nn

from deeplearning_python.models import ModelConfig, SIMPLE_MLP_PRESETS


def test_model_config_defaults_to_baseline_preset() -> None:
    config = ModelConfig()
    hidden_sizes, dropout = SIMPLE_MLP_PRESETS["baseline"]
    assert config.preset == "baseline"
    assert config.hidden_sizes == hidden_sizes
    assert config.dropout == dropout


def test_model_config_applies_named_preset_and_allows_hidden_override() -> None:
    config = ModelConfig(preset="wide_dropout", hidden_sizes=(32, 16))
    preset_hidden, preset_dropout = SIMPLE_MLP_PRESETS["wide_dropout"]
    assert config.hidden_sizes == (32, 16)
    assert config.dropout == preset_dropout
    assert preset_hidden != config.hidden_sizes


def test_model_config_allows_explicit_dropout_override() -> None:
    config = ModelConfig(preset="wide_dropout", dropout=0.25)
    preset_hidden, _ = SIMPLE_MLP_PRESETS["wide_dropout"]
    assert config.hidden_sizes == preset_hidden
    assert pytest.approx(config.dropout) == 0.25


def test_model_config_supports_manual_configuration_without_preset() -> None:
    config = ModelConfig(hidden_sizes=(32,), dropout=0.15)
    assert config.preset is None
    assert config.hidden_sizes == (32,)
    assert pytest.approx(config.dropout) == 0.15


def test_model_config_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown simple MLP preset"):
        ModelConfig(preset="mystery")


def test_model_config_rejects_presets_for_non_simple() -> None:
    with pytest.raises(ValueError, match="only supported for the 'simple'"):
        ModelConfig(name="regularized", preset="baseline")


def test_create_model_uses_preset_dropout() -> None:
    config = ModelConfig(preset="wide_dropout")
    model = config.create_model()
    dropouts = [module for module in model.network if isinstance(module, nn.Dropout)]
    assert dropouts, "expected Dropout layers when preset specifies non-zero dropout"
    assert all(module.p == SIMPLE_MLP_PRESETS["wide_dropout"][1] for module in dropouts)


def test_create_model_respects_explicit_dropout_override() -> None:
    config = ModelConfig(preset="wide_dropout", dropout=0.4)
    model = config.create_model()
    dropouts = [module for module in model.network if isinstance(module, nn.Dropout)]
    assert dropouts
    assert all(module.p == pytest.approx(0.4) for module in dropouts)
