"""Evaluation metrics and result tracking for [[ package_name ]].

Provides structured result storage with Pydantic models and aggregation
utilities for multi-seed experiment analysis.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from [[ package_name ]].infra.storage import pull, push


class EvalResult(BaseModel):
    """Structured evaluation result for a single run.

    This model standardizes evaluation output, making it easy to:
    - Save/load results consistently
    - Aggregate across multiple seeds or runs
    - Compare different evaluation configurations

    Attributes:
        run_id: Unique identifier for the training run.
        eval_name: Name of the evaluation (e.g., "test_set", "ood_generalization").
        timestamp: When the evaluation was performed.
        metrics: Dictionary of metric names to values.
        config: Evaluation configuration used.
    """

    run_id: str = Field(description="Unique run identifier")
    eval_name: str = Field(description="Name of this evaluation")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp of evaluation",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Computed metrics",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation configuration",
    )

    def save(self, path: str) -> None:
        """Save the evaluation result to storage as JSON.

        Args:
            path: Storage path for the JSON file.
        """
        # Serialize to JSON string first
        json_str = self.model_dump_json(indent=2)
        # Write to local temp, then push to storage
        local_path = f"/tmp/{Path(path).name}"
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w") as f:
            f.write(json_str)
        push(local_path, path)

    @classmethod
    def load(cls, path: str) -> EvalResult:
        """Load an evaluation result from storage.

        Args:
            path: Storage path to the JSON file.

        Returns:
            Parsed EvalResult instance.
        """
        local_path = f"/tmp/{Path(path).name}"
        pull(path, local_path)
        with open(local_path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return (
            f"EvalResult(run_id='{self.run_id}', eval_name='{self.eval_name}', "
            f"metrics={self.metrics})"
        )


def aggregate(results: list[EvalResult]) -> dict[str, dict[str, float]]:
    """Aggregate evaluation results across multiple runs.

    Computes mean and standard deviation for all metrics that appear
    in every result. Useful for multi-seed experiments.

    Args:
        results: List of EvalResult instances to aggregate.

    Returns:
        Dictionary mapping metric names to {"mean": float, "std": float}.

    Raises:
        ValueError: If the results list is empty or metrics don't match.
    """
    if not results:
        raise ValueError("Cannot aggregate empty list of results")

    if len(results) == 1:
        return {name: {"mean": value, "std": 0.0} for name, value in results[0].metrics.items()}

    # Get the intersection of all metric names
    all_keys = [set(r.metrics.keys()) for r in results]
    common_keys = all_keys[0].intersection(*all_keys[1:])

    if not common_keys:
        raise ValueError("No common metrics found across results")

    aggregated: dict[str, dict[str, float]] = {}
    for key in sorted(common_keys):
        values = [r.metrics[key] for r in results]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated[key] = {"mean": mean, "std": std}

    return aggregated


def compare_runs(baseline: EvalResult, treatment: EvalResult) -> dict[str, dict[str, float]]:
    """Compare two evaluation runs and compute deltas.

    Args:
        baseline: The baseline (control) evaluation result.
        treatment: The treatment (experiment) evaluation result.

    Returns:
        Dictionary of metric names to dicts with "baseline", "treatment",
        "delta", and "delta_pct" float values.
    """
    comparison: dict[str, dict[str, float]] = {}
    all_keys = set(baseline.metrics.keys()) | set(treatment.metrics.keys())

    for key in sorted(all_keys):
        b = baseline.metrics.get(key, float("nan"))
        t = treatment.metrics.get(key, float("nan"))
        delta = t - b
        delta_pct = (delta / b * 100) if b != 0 else float("inf") if delta != 0 else 0.0
        comparison[key] = {
            "baseline": b,
            "treatment": t,
            "delta": delta,
            "delta_pct": delta_pct,
        }

    return comparison
