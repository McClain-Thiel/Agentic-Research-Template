"""Base model class with configuration-driven initialization and storage integration.

All model implementations should subclass BaseModel and override the domain-specific
methods marked with TODO comments.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

from [[ package_name ]].config.experiment import ModelConfig
from [[ package_name ]].infra.storage import pull, push


class BaseModel(nn.Module):
    """Base model class for all architectures in [[ package_name ]].

    Provides configuration-driven construction, save/load via fsspec-backed
    storage, and a pattern for domain-specific model implementations.

    TODO: Create domain-specific subclasses (e.g., TransformerModel, CNNModel)
    that implement the actual forward pass and architecture logic.

    Args:
        cfg: Model configuration specifying architecture hyperparameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # TODO: Define domain-specific layers here
        # Example for a Transformer:
        #   self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        #   encoder_layer = nn.TransformerEncoderLayer(
        #       d_model=cfg.hidden_size, nhead=cfg.num_heads, batch_first=True
        #   )
        #   self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        #   self.head = nn.Linear(cfg.hidden_size, cfg.num_classes)
        self._dummy = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        TODO: Replace with actual domain-specific forward logic.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor. Shape depends on the task.
        """
        # Stub forward pass for testing
        return self._dummy(x)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> BaseModel:
        """Create a model instance from a configuration.

        This factory method is the preferred way to instantiate models
        as it ensures configuration consistency.

        Args:
            cfg: Model configuration.

        Returns:
            Instantiated model (may be a subclass based on cfg.name).
        """
        logger.info(f"Creating model '{cfg.name}' with config: {cfg.model_dump()}")
        # TODO: Add branching logic here to instantiate different model classes
        # based on cfg.name (e.g., "transformer" -> TransformerModel)
        # Example:
        #   model_registry = {
        #       "transformer": TransformerModel,
        #       "cnn": CNNModel,
        #       "mlp": MLPModel,
        #   }
        #   model_cls = model_registry.get(cfg.name, cls)
        #   return model_cls(cfg)
        return cls(cfg)

    def save(self, path: str) -> None:
        """Save model checkpoint to storage.

        The checkpoint includes model state dict and configuration,
        allowing full reconstruction with load().

        Args:
            path: Storage path for the checkpoint.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.cfg.model_dump(),
            "version": "0.1.0",
        }
        # Save to local temp first, then push to storage
        local_path = f"/tmp/{Path(path).name}"
        torch.save(checkpoint, local_path)
        push(local_path, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, cfg: ModelConfig) -> BaseModel:
        """Load a model checkpoint from storage.

        Args:
            path: Storage path to the checkpoint.
            cfg: Model configuration (used to construct the model architecture).

        Returns:
            Loaded model with restored weights.
        """
        local_path = f"/tmp/{Path(path).name}"
        pull(path, local_path)
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)
        model = cls.from_config(cfg)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Model loaded from {path}")
        return model

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters.

        Returns:
            Dictionary with 'total' and 'trainable' parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def summary(self) -> str:
        """Return a human-readable model summary.

        Returns:
            Formatted string with model architecture overview.
        """
        counts = self.count_parameters()
        lines = [
            f"Model: {self.__class__.__name__}",
            f"  Config: {self.cfg.name}",
            f"  Total params: {counts['total']:,}",
            f"  Trainable params: {counts['trainable']:,}",
        ]
        return "\n".join(lines)
