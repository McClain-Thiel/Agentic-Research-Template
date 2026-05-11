"""Tests for data loading utilities."""

from __future__ import annotations

import pytest
import torch

from [[ package_name ]].config.experiment import DataConfig
from [[ package_name ]].data.loaders import StubDataset, get_dataloader


class TestStubDataset:
    """Tests for the StubDataset synthetic data."""

    def _make_config(self, **kwargs) -> DataConfig:
        defaults = {"batch_size": 4, "num_workers": 0}
        defaults.update(kwargs)
        return DataConfig(**defaults)

    def test_dataset_has_correct_length(self) -> None:
        """StubDataset should report the correct length."""
        cfg = self._make_config()
        dataset = StubDataset(cfg, split="train", num_samples=50)

        assert len(dataset) == 50

    def test_sample_has_required_keys(self) -> None:
        """Each sample should contain expected keys."""
        cfg = self._make_config()
        dataset = StubDataset(cfg, split="train", num_samples=10)
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "input" in sample
        assert "label" in sample

    def test_sample_tensors_have_correct_dtype(self) -> None:
        """Sample values should be torch tensors."""
        cfg = self._make_config()
        dataset = StubDataset(cfg, split="train", num_samples=10)
        sample = dataset[0]

        assert isinstance(sample["input"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)


class TestGetDataLoader:
    """Tests for the get_dataloader factory function."""

    def _make_config(self, **kwargs) -> DataConfig:
        defaults = {"batch_size": 4, "num_workers": 0}
        defaults.update(kwargs)
        return DataConfig(**defaults)

    def test_returns_dataloader(self) -> None:
        """get_dataloader should return a PyTorch DataLoader."""
        cfg = self._make_config()
        loader = get_dataloader(cfg, split="train")

        from torch.utils.data import DataLoader

        assert isinstance(loader, DataLoader)

    def test_dataloader_batch_size(self) -> None:
        """DataLoader should use the configured batch size."""
        cfg = self._make_config(batch_size=8)
        loader = get_dataloader(cfg, split="train")

        batch = next(iter(loader))
        assert batch["input"].shape[0] == 8

    def test_dataloader_batch_has_required_keys(self) -> None:
        """Each batch should contain expected keys."""
        cfg = self._make_config()
        loader = get_dataloader(cfg, split="train")

        batch = next(iter(loader))
        assert isinstance(batch, dict)
        assert "input" in batch
        assert "label" in batch

    def test_dataloader_batch_tensors(self) -> None:
        """Batch values should be torch tensors."""
        cfg = self._make_config()
        loader = get_dataloader(cfg, split="train")

        batch = next(iter(loader))
        assert isinstance(batch["input"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)

    def test_train_val_test_splits(self) -> None:
        """Should create loaders for all three splits."""
        cfg = self._make_config()

        train_loader = get_dataloader(cfg, split="train")
        val_loader = get_dataloader(cfg, split="val")
        test_loader = get_dataloader(cfg, split="test")

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_invalid_split_raises(self) -> None:
        """An invalid split name should raise ValueError."""
        cfg = self._make_config()

        with pytest.raises(ValueError):
            get_dataloader(cfg, split="invalid_split")
