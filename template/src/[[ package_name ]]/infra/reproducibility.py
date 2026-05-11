"""Reproducibility utilities: seed setting, git tracking, environment logging.

This module provides helpers for deterministic experiments:

- :func:`set_seeds` — configure all pseudo-random number generators
- :func:`get_git_hash` — record the exact code version used
- :func:`check_clean_repo` — guard against uncommitted changes
- :func:`log_environment` — persist runtime metadata to W&B

Example::

    from [[ package_name ]].infra.reproducibility import set_seeds, get_git_hash, check_clean_repo

    set_seeds(42)
    check_clean_repo(strict=True)  # fail fast if code is dirty
    print(f"Running commit {get_git_hash()}")
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Seeds are applied to:

    * Python :mod:`random`
    * NumPy (if installed)
    * PyTorch CPU and CUDA
    * PyTorch backends (cudnn, deterministic flags)

    Args:
        seed: Integer seed value (should be non-negative).

    Raises:
        ValueError: If *seed* is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
        logger.debug("Set NumPy random seed to {}", seed)
    except ImportError:
        logger.debug("NumPy not installed — skipping numpy seed")

    # PyTorch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic behaviour
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.debug("Set PyTorch random seed to {} (all devices)", seed)
    except ImportError:
        logger.debug("PyTorch not installed — skipping torch seed")

    logger.info("Global random seed set to {}", seed)


# ---------------------------------------------------------------------------
# Git tracking
# ---------------------------------------------------------------------------


def get_git_hash() -> str:
    """Get the current git commit hash.

    Returns:
        Short git hash (7 hex characters), or ``"unknown"`` if the current
        working directory is not inside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def check_clean_repo(strict: bool = False) -> None:
    """Check if the git repository has uncommitted changes.

    A dirty repository can lead to unreproducible results because the
    exact code version is not recorded.

    Args:
        strict: If ``True``, raise :exc:`RuntimeError` when uncommitted
            changes are detected.  If ``False``, log a warning and continue.

    Raises:
        RuntimeError: If *strict* is ``True`` and uncommitted changes exist.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        is_dirty = bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Unable to determine git status — not a git repo or git not installed")
        return

    if is_dirty:
        msg = (
            "Git repository has uncommitted changes. "
            "Commit or stash changes for full reproducibility."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.debug("Git repository is clean")


# ---------------------------------------------------------------------------
# Environment logging
# ---------------------------------------------------------------------------


def log_environment(run: Any) -> None:
    """Log environment information to a W&B run.

    Records the git commit hash, Python version, CUDA availability,
    and key dependency versions.  Safe to call even when optional
    dependencies are missing.

    Args:
        run: An active :class:`wandb.Run` instance (or any object with
            a ``config`` attribute that supports dict-style assignment).
    """
    git_hash = get_git_hash()
    python_version = sys.version.replace("\n", " ")

    env_info: dict[str, str] = {
        "git_hash": git_hash,
        "python_version": python_version,
    }

    # CUDA info
    try:
        import torch

        env_info["torch"] = torch.__version__
        env_info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda or "unknown"
            env_info["gpu_count"] = str(torch.cuda.device_count())
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env_info["torch"] = "not_installed"

    # NumPy
    try:
        import numpy as np

        env_info["numpy"] = np.__version__
    except ImportError:
        env_info["numpy"] = "not_installed"

    # Key ML libraries
    for pkg in ("transformers", "datasets", "accelerate", "peft"):
        try:
            mod = __import__(pkg)
            env_info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env_info[pkg] = "not_installed"

    # Platform
    try:
        import platform

        env_info["platform"] = platform.platform()
        env_info["processor"] = platform.processor() or "unknown"
    except ImportError:
        pass

    # Push to W&B
    try:
        run.config.update(env_info)
        logger.info(
            "Logged environment to W&B — git={}, python={}, torch={}",
            git_hash,
            python_version.split()[0],
            env_info.get("torch", "?"),
        )
    except Exception as exc:
        logger.warning("Failed to log environment to W&B: {}", exc)
