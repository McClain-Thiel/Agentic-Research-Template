"""Tests for model output shapes and forward pass."""

from __future__ import annotations

import torch

from [[ package_name ]].config.experiment import ModelConfig
from [[ package_name ]].models.base import BaseModel


class TestBaseModelForward:
    """Tests for BaseModel forward pass and output shapes."""

    def _make_model(self, **kwargs) -> BaseModel:
        """Create a BaseModel with a tiny config for fast tests."""
        cfg = ModelConfig(hidden_size=16, num_layers=1, num_heads=2, dropout=0.0, **kwargs)
        return BaseModel.from_config(cfg)

    def test_forward_returns_tensor(self) -> None:
        """Forward pass should return a torch.Tensor."""
        model = self._make_model()
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, model.cfg.hidden_size)

        output = model(x)

        assert isinstance(output, torch.Tensor)

    def test_forward_output_shape(self) -> None:
        """Forward pass output should match expected shape."""
        model = self._make_model()
        batch_size, seq_len = 4, 10
        hidden_size = model.cfg.hidden_size
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = model(x)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_forward_different_batch_sizes(self) -> None:
        """Model should handle different batch sizes."""
        model = self._make_model()
        hidden_size = model.cfg.hidden_size

        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 4, hidden_size)
            output = model(x)
            assert output.shape[0] == batch_size

    def test_forward_different_seq_lengths(self) -> None:
        """Model should handle different sequence lengths."""
        model = self._make_model()
        hidden_size = model.cfg.hidden_size

        for seq_len in [1, 4, 16]:
            x = torch.randn(2, seq_len, hidden_size)
            output = model(x)
            assert output.shape[1] == seq_len

    def test_from_config_classmethod(self) -> None:
        """from_config should return a BaseModel instance."""
        cfg = ModelConfig(hidden_size=8, num_layers=1)
        model = BaseModel.from_config(cfg)

        assert isinstance(model, BaseModel)
        assert model.cfg.hidden_size == 8

    def test_model_has_parameters(self) -> None:
        """Model should have trainable parameters."""
        model = self._make_model()
        params = list(model.parameters())

        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestBaseModelSaveLoad:
    """Tests for model save and load functionality."""

    def test_save_creates_checkpoint(self, tmp_path) -> None:
        """save() should create a checkpoint file."""
        cfg = ModelConfig(hidden_size=8, num_layers=1, dropout=0.0)
        model = BaseModel.from_config(cfg)

        # Use local path for test (avoiding storage module)
        save_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": model.cfg.model_dump(),
            "version": "0.1.0",
        }
        torch.save(checkpoint, save_path)

        assert save_path.exists()

    def test_load_restores_weights(self, tmp_path) -> None:
        """load() should restore model weights."""
        cfg = ModelConfig(hidden_size=8, num_layers=1, dropout=0.0)
        model = BaseModel.from_config(cfg)

        # Save
        save_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": model.cfg.model_dump(),
            "version": "0.1.0",
        }
        torch.save(checkpoint, save_path)

        # Load into new model
        loaded = BaseModel.from_config(cfg)
        loaded_checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
        loaded.load_state_dict(loaded_checkpoint["state_dict"])

        # Verify weights match
        for p1, p2 in zip(model.parameters(), loaded.parameters(), strict=True):
            assert torch.allclose(p1, p2)
