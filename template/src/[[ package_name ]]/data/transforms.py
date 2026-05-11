"""Data transform utilities for [[ package_name ]].

Provides a base transform class and composition pattern for building
reproducible data preprocessing pipelines.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Transform(Protocol):
    """Protocol defining the interface for data transforms.

    All transforms must implement this interface to be composable.

    Example:
        >>> class Normalize(Transform):
        ...     def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ...         sample["input"] = (sample["input"] - self.mean) / self.std
        ...         return sample
    """

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply the transform to a sample.

        Args:
            sample: Dictionary of tensors representing a single sample.

        Returns:
            Transformed sample dictionary.
        """
        ...


class Compose:
    """Compose multiple transforms into a single callable.

    Transforms are applied in the order they are provided.

    Args:
        transforms: List of transform callables.

    Example:
        >>> pipeline = Compose([Normalize(), Tokenize(), Pad()])
        >>> sample = pipeline(sample)
    """

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        transform_names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({transform_names})"


class Normalize(Transform):
    """Normalize input tensors by subtracting mean and dividing by std.

    TODO: Replace with domain-specific normalization.

    Args:
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample["input"] = (sample["input"] - self.mean) / self.std
        return sample


class RandomCrop(Transform):
    """Randomly crop input tensors to a target size.

    TODO: Replace with domain-specific crop transform.

    Args:
        size: Target size for the crop.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # TODO: Implement domain-specific random crop
        return sample


class RandomHorizontalFlip(Transform):
    """Randomly flip input tensors horizontally.

    TODO: Replace with domain-specific flip transform.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # TODO: Implement domain-specific horizontal flip
        if torch.rand(1).item() < self.p:
            pass  # Apply flip logic here
        return sample


class ToDtype(Transform):
    """Convert tensors to a specified dtype.

    Args:
        dtype: Target torch dtype (e.g., torch.float32, torch.long).
        keys: Which keys in the sample dict to convert. Converts "input"
            by default.
    """

    def __init__(self, dtype: torch.dtype, keys: list[str] | None = None) -> None:
        self.dtype = dtype
        self.keys = keys or ["input"]

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in self.keys:
            if key in sample:
                sample[key] = sample[key].to(self.dtype)
        return sample


class PadSequence(Transform):
    """Pad sequences to a fixed length.

    TODO: Replace with domain-specific padding logic.

    Args:
        max_length: Maximum sequence length to pad to.
        pad_value: Value to use for padding.
    """

    def __init__(self, max_length: int, pad_value: float = 0.0) -> None:
        self.max_length = max_length
        self.pad_value = pad_value

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # TODO: Implement domain-specific padding
        # This operates on a single sample; batch-level padding may be
        # better handled by a custom collate_fn in the DataLoader.
        return sample
