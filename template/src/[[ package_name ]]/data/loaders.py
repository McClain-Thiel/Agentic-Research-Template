"""Data loading utilities with fsspec and HuggingFace datasets integration.

This module provides a DataConfig-aware factory for creating PyTorch DataLoaders.
All file I/O goes through the storage abstraction — never use raw open() for data files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from [[ package_name ]].config.experiment import DataConfig
from [[ package_name ]].infra.storage import storage_path


class StubDataset(Dataset):
    """A minimal synthetic dataset for testing and development.

    TODO: Replace with your actual dataset implementation. This stub generates
    random tensors matching the expected structure for quick iteration.

    Args:
        cfg: Data configuration.
        split: Dataset split ("train", "val", or "test").
        num_samples: Number of synthetic samples to generate.
    """

    def __init__(self, cfg: DataConfig, split: str, num_samples: int = 100) -> None:
        self.cfg = cfg
        self.split = split
        self.num_samples = num_samples
        logger.info(
            f"Initialized StubDataset(split='{split}', num_samples={num_samples})"
        )
        # TODO: Replace with actual dataset initialization.
        # Load data indices, metadata, or file references here.
        # Use storage_path() for any file I/O, never raw open().

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample.

        TODO: Replace with actual data loading logic.
        Expected return format: {"input": tensor, "label": tensor, "mask": tensor}
        """
        # Stub: generate random tensors matching expected shape
        # TODO: Replace with actual data loading from storage via storage_path()
        seq_len = getattr(self.cfg, "seq_len", 128)
        hidden_size = getattr(self.cfg, "hidden_size", 256)

        return {
            "input": torch.randn(seq_len, hidden_size),
            "label": torch.randint(0, 10, (1,)).squeeze(),
            "mask": torch.ones(seq_len, dtype=torch.bool),
        }


def _load_hf_dataset(
    cfg: DataConfig, split: str
) -> Dataset:
    """Load a dataset from the HuggingFace Hub.

    TODO: Implement HuggingFace datasets integration for your domain.

    Args:
        cfg: Data configuration with HuggingFace dataset info.
        split: Dataset split to load.

    Returns:
        A PyTorch-compatible Dataset.
    """
    logger.info(f"Loading HuggingFace dataset: {cfg.dataset_name} (split={split})")
    # TODO: Add HuggingFace datasets integration here
    # Example:
    #   from datasets import load_dataset
    #   hf_ds = load_dataset(cfg.dataset_name, split=split)
    #   return hf_ds.with_format("torch")
    raise NotImplementedError("HuggingFace dataset loading not yet implemented — add your dataset logic here.")


def _load_local_dataset(
    cfg: DataConfig, split: str
) -> Dataset:
    """Load a dataset from local or remote storage.

    TODO: Implement local/remote dataset loading for your domain.
    Uses the storage module for all file I/O.

    Args:
        cfg: Data configuration with data path info.
        split: Dataset split to load.

    Returns:
        A PyTorch-compatible Dataset.
    """
    data_path = storage_path(cfg.data_path) / split
    logger.info(f"Loading local dataset from: {data_path}")
    # TODO: Add local dataset loading logic here
    # Use storage_path() for all file paths, never raw open()
    raise NotImplementedError("Local dataset loading not yet implemented — add your dataset logic here.")


def _build_collate_fn(cfg: DataConfig):
    """Build a collate function for batching samples.

    TODO: Implement domain-specific batching logic (e.g., padding, packing).

    Args:
        cfg: Data configuration.

    Returns:
        Collate function for DataLoader.
    """
    # TODO: Replace with domain-specific collate function
    # Example: pad sequences to max length in batch
    return None


def get_dataloader(cfg: DataConfig, split: str) -> DataLoader:
    """Create a DataLoader for the specified split.

    This factory function creates a DataLoader configured according to the
    experiment's DataConfig. The actual dataset implementation is domain-specific
    and should be filled in where TODO comments are marked.

    Args:
        cfg: Data configuration from the experiment config.
        split: Which split to load ("train", "val", or "test").

    Returns:
        Configured PyTorch DataLoader.

    Raises:
        ValueError: If split is not one of "train", "val", or "test".
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    logger.info(f"Creating DataLoader for split='{split}'")

    # TODO: Select dataset source based on config
    # Option 1: HuggingFace datasets
    # if cfg.source == "huggingface":
    #     dataset = _load_hf_dataset(cfg, split)
    #
    # Option 2: Local or remote storage (fsspec-backed)
    # elif cfg.source == "local":
    #     dataset = _load_local_dataset(cfg, split)
    #
    # Option 3: Stub dataset for quick iteration (default)
    # else:
    #     dataset = StubDataset(cfg, split)

    dataset: Dataset = StubDataset(cfg, split)

    # Build sampler for distributed training
    # TODO: Enable DistributedSampler when running multi-GPU
    sampler = None
    # if cfg.distributed and split == "train":
    #     sampler = DistributedSampler(dataset, shuffle=cfg.shuffle)

    shuffle = (split == "train") and cfg.shuffle if hasattr(cfg, "shuffle") else (split == "train")

    # Build collate function for domain-specific batching
    collate_fn = _build_collate_fn(cfg)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(cfg, "num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=getattr(cfg, "pin_memory", True),
        drop_last=(split == "train"),
        prefetch_factor=getattr(cfg, "prefetch_factor", 2) if getattr(cfg, "num_workers", 0) > 0 else None,
    )

    logger.info(
        f"DataLoader created: {len(loader)} batches "
        f"(batch_size={cfg.batch_size}, shuffle={shuffle})"
    )
    return loader
