"""Weights & Biases logging wrapper with rich console output.

This module provides a thin, typed abstraction over the ``wandb`` SDK.
All functions gracefully degrade when W&B credentials are not configured
(see :data:`settings.WANDB_API_KEY`).

Example::

    from [[ package_name ]].config.experiment import ExperimentConfig
    from [[ package_name ]].infra.logging import init_run, log_metrics, finish_run

    cfg = ExperimentConfig(name="my_experiment")
    run = init_run(cfg)
    log_metrics({"loss": 0.42, "accuracy": 0.91}, step=100)
    finish_run(run)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.console import Console

from [[ package_name ]].config.experiment import ExperimentConfig
from [[ package_name ]].config.settings import settings
from [[ package_name ]].infra.reproducibility import get_git_hash

if TYPE_CHECKING:
    import wandb.sdk.wandb_run

console = Console()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_wandb_configured() -> bool:
    """Return whether W&B API key is available."""
    return bool(settings.WANDB_API_KEY)


def _mask_api_key(key: str) -> str:
    """Return a masked representation of an API key for safe logging."""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init_run(
    cfg: ExperimentConfig,
    run_id: str | None = None,
) -> wandb.sdk.wandb_run.Run | Any:
    """Initialize a Weights & Biases run for experiment tracking.

    The full experiment configuration is serialised and uploaded as the
    run configuration.  The git commit hash is automatically attached as
    a tag.

    Args:
        cfg: Full experiment configuration.
        run_id: Optional existing run ID to resume a previous run.

    Returns:
        The initialised :class:`wandb.Run` instance, or a no-op stub
        when W&B is not configured.
    """
    import wandb

    if not _is_wandb_configured():
        logger.warning(
            "W&B API key not set (WANDB_API_KEY). "
            "Tracking is disabled — set the key in .env to enable."
        )
        console.print("[yellow]⚠ W&B not configured — running without experiment tracking[/yellow]")
        return _NoOpRun()

    git_hash = get_git_hash()
    tags = list(cfg.tags)
    if git_hash != "unknown":
        tags.append(f"git:{git_hash}")

    wandb.login(key=settings.WANDB_API_KEY)

    run = wandb.init(
        project=settings.WANDB_PROJECT,
        entity=settings.WANDB_ENTITY or None,
        name=cfg.name,
        id=run_id,
        resume="must" if run_id else None,
        tags=tags,
        notes=cfg.description or None,
        config=cfg.model_dump(mode="json"),
    )

    console.print(
        f"[bold green]✓ W&B run initialised[/bold green] "
        f"[link={run.url}]{run.id}[/link] — {cfg.name}"
    )
    logger.info(
        "W&B run started — id={}, project={}, git={}",
        run.id,
        settings.WANDB_PROJECT,
        git_hash,
    )

    return run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the current W&B run.

    Args:
        metrics: Dictionary mapping metric names to scalar values.
        step: Optional global step number.  If omitted, W&B uses its
            internal step counter.
    """
    import wandb

    if not _is_wandb_configured():
        # Print to console as a fallback
        step_str = f" (step {step})" if step is not None else ""
        for key, value in metrics.items():
            console.print(f"  [dim]{key}[/dim] = {value:.6f}{step_str}")
        return

    if wandb.run is None:
        logger.warning("log_metrics called but no W&B run is active")
        return

    wandb.log(metrics, step=step)
    logger.debug("Logged metrics at step {}: {}", step, metrics)


def log_artifact(path: str, name: str, artifact_type: str) -> None:
    """Log a file or directory as a W&B artifact.

    Args:
        path: Local path to the artifact (file or directory).
        name: Name for the artifact (used as the artifact identifier).
        artifact_type: W&B artifact type — one of ``"model"``, ``"dataset"``,
            ``"result"``, ``"code"``, etc.
    """
    import wandb

    if not _is_wandb_configured():
        logger.debug(
            "Skipping artifact upload (W&B not configured): {} → {}",
            path,
            name,
        )
        return

    if wandb.run is None:
        logger.warning("log_artifact called but no W&B run is active")
        return

    artifact_path = Path(path)
    if not artifact_path.exists():
        logger.warning("Artifact path does not exist: {}", path)
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)

    if artifact_path.is_dir():
        artifact.add_dir(str(artifact_path))
    else:
        artifact.add_file(str(artifact_path))

    wandb.log_artifact(artifact)
    logger.info("Logged artifact '{}' (type={}) from {}", name, artifact_type, path)


def finish_run(run: wandb.sdk.wandb_run.Run | Any, status: str = "success") -> None:
    """Finalize a W&B run.

    Args:
        run: The :class:`wandb.Run` instance to finish (or a no-op stub).
        status: Run status string — ``"success"``, ``"failed"``, ``"killed"``,
            etc.  Recorded in W&B for filtering.
    """

    if isinstance(run, _NoOpRun):
        console.print("[dim]W&B not configured — skipping finish[/dim]")
        return

    try:
        # Mark the run with its final status
        if hasattr(run, "config") and run.config is not None:
            run.config["run_status"] = status

        run.finish()
        console.print(f"[bold green]✓ W&B run finished[/bold green] — status={status}")
        logger.info("W&B run finished — id={}, status={}", run.id, status)
    except Exception as exc:
        logger.error("Error finishing W&B run: {}", exc)
        console.print(f"[red]Error finishing W&B run: {exc}[/red]")


# ---------------------------------------------------------------------------
# No-op stub for when W&B is not configured
# ---------------------------------------------------------------------------


class _NoOpRun:
    """Stand-in for :class:`wandb.Run` when tracking is disabled.

    All attribute accesses and method calls are silently ignored,
    allowing downstream code to use ``run.config["foo"] = bar``
    without checking whether W&B is active.
    """

    id: str = "local"
    url: str = ""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

    def log(self, *args: Any, **kwargs: Any) -> None:
        """No-op replacement for :meth:`wandb.Run.log`."""

    def finish(self) -> None:
        """No-op replacement for :meth:`wandb.Run.finish`."""
