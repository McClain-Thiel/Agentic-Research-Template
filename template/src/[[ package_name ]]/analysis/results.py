"""Result loading and experiment registry access for [[ package_name ]].

Provides utilities for reading evaluation results, accessing the experiment
registry, and retrieving run configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from [[ package_name ]].analysis.metrics import EvalResult
from [[ package_name ]].config.experiment import ExperimentConfig
from [[ package_name ]].infra.storage import pull, storage_path


def load_results(run_id: str) -> list[EvalResult]:
    """Load all evaluation results for a given run.

    Reads all JSON files from results/{run_id}/ and parses them as EvalResult.

    Args:
        run_id: The run identifier.

    Returns:
        List of EvalResult instances.
    """
    results_dir = storage_path(f"results/{run_id}")
    logger.info(f"Loading results from: {results_dir}")

    # TODO: List files in results directory via storage abstraction
    # For now, attempt to read known result files
    eval_files: list[Path] = []
    if results_dir.exists():
        eval_files = sorted(results_dir.glob("*.json"))

    if not eval_files:
        logger.warning(f"No result files found in {results_dir}")
        return []

    results: list[EvalResult] = []
    for eval_file in eval_files:
        try:
            result = EvalResult.load(str(eval_file))
            results.append(result)
            logger.debug(f"Loaded result: {eval_file.name}")
        except Exception:
            logger.warning(f"Failed to load result file: {eval_file}")
            continue

    logger.info(f"Loaded {len(results)} evaluation results for run '{run_id}'")
    return results


def load_registry() -> list[dict[str, Any]]:
    """Load the experiment registry.

    Returns:
        List of experiment entries from experiments/registry.yaml.
    """
    registry_path = storage_path("experiments/registry.yaml")
    logger.info(f"Loading experiment registry from: {registry_path}")

    # Pull from storage to local temp
    local_path = f"/tmp/registry_{os.getpid()}.yaml"
    try:
        pull(str(registry_path), local_path)
    except Exception:
        logger.warning("Registry not found in storage, trying local path")
        local_path = str(registry_path)

    if not Path(local_path).exists():
        logger.warning("No registry file found — returning empty list")
        return []

    with open(local_path) as f:
        registry = yaml.safe_load(f)

    if registry is None:
        return []

    experiments = registry if isinstance(registry, list) else registry.get("experiments", [])
    logger.info(f"Loaded {len(experiments)} experiments from registry")
    return experiments


def get_run_config(run_id: str) -> ExperimentConfig:
    """Retrieve the experiment configuration for a specific run.

    Attempts to load the config from storage first, then falls back
    to local files.

    Args:
        run_id: The run identifier.

    Returns:
        The experiment configuration used for this run.
    """
    # Try storage path first
    config_path = f"results/{run_id}/config.yaml"
    local_path = f"/tmp/config_{run_id}.yaml"

    try:
        pull(config_path, local_path)
        logger.info(f"Loaded config from storage: {config_path}")
        return ExperimentConfig.from_yaml(local_path)
    except Exception:
        logger.info("Config not found in storage, trying local path")

    # Fallback: try local experiments directory
    local_config = Path(f"experiments/{run_id}/config.yaml")
    if local_config.exists():
        logger.info(f"Loaded config from local path: {local_config}")
        return ExperimentConfig.from_yaml(str(local_config))

    # Final fallback: try to find in registry
    registry = load_registry()
    for entry in registry:
        if entry.get("run_id") == run_id or entry.get("name") == run_id:
            config_file = entry.get("config_file")
            if config_file:
                return ExperimentConfig.from_yaml(config_file)

    raise FileNotFoundError(f"Could not find configuration for run '{run_id}'")


def list_runs(experiment_name: str | None = None) -> list[dict[str, Any]]:
    """List all recorded runs, optionally filtered by experiment name.

    Args:
        experiment_name: Optional experiment name to filter by.

    Returns:
        List of run metadata dictionaries.
    """
    registry = load_registry()
    runs = []
    for entry in registry:
        if experiment_name is None or entry.get("name") == experiment_name:
            runs.append(entry)
    return runs
