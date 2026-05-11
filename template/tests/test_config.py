"""Tests for experiment configuration models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from [[ package_name ]].config.experiment import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)

MINIMAL_CONFIG_YAML = """
name: test_experiment
description: A minimal test experiment
seed: 123
model:
  name: test_model
  hidden_size: 64
  num_layers: 2
data:
  dataset: dummy
  batch_size: 16
training:
  epochs: 2
  lr: 0.001
eval:
  metrics: [accuracy]
"""


def _write_temp_yaml(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        return Path(f.name)


class TestExperimentConfigLoad:
    """Tests for loading ExperimentConfig from YAML."""

    def test_loads_minimal_config(self) -> None:
        """A minimal valid YAML config should load without errors."""
        path = _write_temp_yaml(MINIMAL_CONFIG_YAML)
        cfg = ExperimentConfig.from_yaml(path)

        assert cfg.name == "test_experiment"
        assert cfg.description == "A minimal test experiment"
        assert cfg.seed == 123
        assert cfg.model.hidden_size == 64
        assert cfg.data.batch_size == 16
        assert cfg.training.epochs == 2
        assert cfg.training.lr == 0.001

    def test_missing_required_field_raises(self) -> None:
        """YAML with invalid structure should raise ValidationError."""
        bad_yaml = "name: test\nmodel:\n  hidden_size: not_a_number"
        path = _write_temp_yaml(bad_yaml)

        with pytest.raises((ValidationError, ValueError)):
            ExperimentConfig.from_yaml(path)

    def test_nonexistent_file_raises(self) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml("/nonexistent/path/config.yaml")


class TestExperimentConfigValidation:
    """Tests for ExperimentConfig field validation."""

    def test_batch_size_must_be_positive(self) -> None:
        """Batch size must be greater than 0."""
        with pytest.raises(ValidationError):
            DataConfig(batch_size=0)

        with pytest.raises(ValidationError):
            DataConfig(batch_size=-1)

    def test_lr_must_be_positive(self) -> None:
        """Learning rate must be greater than 0."""
        with pytest.raises(ValidationError):
            TrainingConfig(lr=0.0)

        with pytest.raises(ValidationError):
            TrainingConfig(lr=-0.001)

    def test_epochs_must_be_positive(self) -> None:
        """Epochs must be at least 1."""
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)

    def test_dropout_must_be_less_than_one(self) -> None:
        """Dropout must be in [0, 1)."""
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.0)

        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.5)

    def test_data_splits_must_sum_to_one(self) -> None:
        """Train/val/test splits must sum to 1.0."""
        with pytest.raises(ValueError):
            DataConfig(train_split=0.5, val_split=0.5, test_split=0.5)

    def test_hidden_size_must_be_even(self) -> None:
        """Hidden size must be even for head splitting."""
        with pytest.raises(ValueError):
            ModelConfig(hidden_size=127)


class TestExperimentConfigRoundTrip:
    """Tests for YAML serialization round-trip."""

    def test_to_from_yaml_roundtrip(self) -> None:
        """Config saved and loaded should be identical."""
        original = ExperimentConfig(
            name="roundtrip_test",
            seed=42,
            model=ModelConfig(hidden_size=256, num_layers=4),
            data=DataConfig(batch_size=32),
            training=TrainingConfig(epochs=10, lr=0.0001),
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        original.to_yaml(temp_path)
        restored = ExperimentConfig.from_yaml(temp_path)

        assert restored.name == original.name
        assert restored.seed == original.seed
        assert restored.model.hidden_size == original.model.hidden_size
        assert restored.model.num_layers == original.model.num_layers
        assert restored.data.batch_size == original.data.batch_size
        assert restored.training.epochs == original.training.epochs
        assert restored.training.lr == original.training.lr

    def test_to_yaml_creates_file(self) -> None:
        """to_yaml should create the output file."""
        cfg = ExperimentConfig(name="file_test")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)
        temp_path.unlink()  # Delete so we can verify creation

        cfg.to_yaml(temp_path)
        assert temp_path.exists()
