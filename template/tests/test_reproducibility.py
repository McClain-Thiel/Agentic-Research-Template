"""Tests for reproducibility utilities."""

from __future__ import annotations

import random

import numpy as np
import torch

from [[ package_name ]].config.experiment import ModelConfig
from [[ package_name ]].infra.reproducibility import get_git_hash, set_seeds
from [[ package_name ]].models.base import BaseModel


class TestSetSeeds:
    """Tests for seed setting and deterministic behavior."""

    def test_same_seed_produces_same_random_numbers(self) -> None:
        """Two runs with the same seed should produce identical random numbers."""
        set_seeds(42)
        random_numbers_1 = [random.random() for _ in range(10)]

        set_seeds(42)
        random_numbers_2 = [random.random() for _ in range(10)]

        assert random_numbers_1 == random_numbers_2

    def test_same_seed_produces_same_numpy_arrays(self) -> None:
        """Two runs with the same seed should produce identical numpy arrays."""
        set_seeds(42)
        array_1 = np.random.randn(10, 10)

        set_seeds(42)
        array_2 = np.random.randn(10, 10)

        np.testing.assert_array_equal(array_1, array_2)

    def test_same_seed_produces_same_torch_tensors(self) -> None:
        """Two runs with the same seed should produce identical torch tensors."""
        set_seeds(42)
        tensor_1 = torch.randn(10, 10)

        set_seeds(42)
        tensor_2 = torch.randn(10, 10)

        assert torch.allclose(tensor_1, tensor_2)

    def test_different_seeds_produce_different_tensors(self) -> None:
        """Two runs with different seeds should produce different tensors."""
        set_seeds(42)
        tensor_1 = torch.randn(10, 10)

        set_seeds(123)
        tensor_2 = torch.randn(10, 10)

        assert not torch.allclose(tensor_1, tensor_2)

    def test_same_seed_produces_same_model_outputs(self) -> None:
        """Two model initializations with the same seed should be identical."""
        cfg = ModelConfig(hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(4, 8, 16)

        set_seeds(42)
        model_1 = BaseModel.from_config(cfg)
        output_1 = model_1(x)

        set_seeds(42)
        model_2 = BaseModel.from_config(cfg)
        output_2 = model_2(x)

        assert torch.allclose(output_1, output_2)

    def test_different_seeds_produce_different_model_outputs(self) -> None:
        """Two model initializations with different seeds should differ."""
        cfg = ModelConfig(hidden_size=16, num_layers=1, dropout=0.0)
        x = torch.randn(4, 8, 16)

        set_seeds(42)
        model_1 = BaseModel.from_config(cfg)
        output_1 = model_1(x)

        set_seeds(123)
        model_2 = BaseModel.from_config(cfg)
        output_2 = model_2(x)

        assert not torch.allclose(output_1, output_2, atol=1e-5)


class TestGetGitHash:
    """Tests for git hash retrieval."""

    def test_returns_string(self) -> None:
        """get_git_hash should always return a string."""
        git_hash = get_git_hash()
        assert isinstance(git_hash, str)

    def test_returns_reasonable_length(self) -> None:
        """Git hash should be a short hash (7 chars) or 'unknown'."""
        git_hash = get_git_hash()
        assert len(git_hash) == 7 or git_hash == "unknown"
