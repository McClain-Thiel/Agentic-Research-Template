"""Main training entry point for [[ package_name ]].

Usage:
    uv run python -m [[ package_name ]].train \\
        --experiment experiments/my_experiment
    uv run python -m [[ package_name ]].train \\
        --experiment experiments/my_experiment --config training.lr=0.001
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from [[ package_name ]].config.experiment import ExperimentConfig
from [[ package_name ]].infra.logging import finish_run, init_run, log_metrics
from [[ package_name ]].infra.reproducibility import check_clean_repo, log_environment, set_seeds
from [[ package_name ]].infra.storage import sync_results
from [[ package_name ]].models.base import BaseModel


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a [[ package_name ]] model")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment folder containing config.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=[],
        help="Config overrides in key=value format (can be used multiple times)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Resume from an existing run ID",
    )
    return parser.parse_args()


def _apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply command-line config overrides to an experiment config.

    Supports dot-notation for nested fields, e.g.:
        training.lr=0.001
        model.hidden_size=512
        training.epochs=10

    Args:
        cfg: Base experiment configuration.
        overrides: List of "key=value" override strings.

    Returns:
        Updated experiment configuration.
    """
    if not overrides:
        return cfg

    # Convert to dict for mutation, then reconstruct
    cfg_dict = cfg.model_dump()

    for override in overrides:
        if "=" not in override:
            logger.warning(f"Skipping invalid override (no '='): {override}")
            continue

        key, value = override.split("=", 1)
        keys = key.split(".")

        # Attempt type inference
        # Try int, then float, then bool, then string
        if value.lower() == "true":
            typed_value: Any = True
        elif value.lower() == "false":
            typed_value = False
        else:
            try:
                typed_value = int(value)
            except ValueError:
                try:
                    typed_value = float(value)
                except ValueError:
                    typed_value = value

        # Navigate to the correct nested dict
        target = cfg_dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = typed_value
        logger.info(f"Config override: {key} = {typed_value}")

    # Reconstruct the config from the updated dict
    return ExperimentConfig.model_validate(cfg_dict)


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Handle SIGTERM by saving checkpoint and syncing results.

    This ensures graceful shutdown on spot instance preemption.
    """
    logger.warning("SIGTERM received — saving checkpoint and syncing results...")
    # TODO: Save checkpoint
    # Example:
    #   if _current_model is not None and _current_cfg is not None:
    #       checkpoint_path = f"checkpoints/{_current_cfg.run_id}/checkpoint.pt"
    #       _current_model.save(checkpoint_path)
    sync_results()
    sys.exit(0)


# Global references for signal handler — populated in main()
_current_model: BaseModel | None = None
_current_cfg: ExperimentConfig | None = None


def main() -> None:
    """Main training loop."""
    args = _parse_args()

    # Register SIGTERM handler for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # 1. Load configuration
    config_path = Path(args.experiment) / "config.yaml"
    cfg = ExperimentConfig.from_yaml(config_path)
    cfg = _apply_overrides(cfg, args.config)
    logger.info(f"Loaded configuration: {cfg.name}")
    logger.info(f"Description: {cfg.description}")

    # 2. Check git repository status
    check_clean_repo(strict=False)

    # 3. Set random seeds for reproducibility
    set_seeds(cfg.seed)
    logger.info(f"Random seeds set to {cfg.seed}")

    # 4. Initialize W&B run
    run = init_run(cfg, run_id=args.run_id)

    # 5. Log environment information
    log_environment(run)

    try:
        # 6. TODO: Training loop (domain-specific)
        # This is where you implement your actual training logic.
        # Example structure:
        #
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = BaseModel.from_config(cfg.model).to(device)
        # train_loader = get_dataloader(cfg.data, "train")
        # val_loader = get_dataloader(cfg.data, "val")
        # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
        #
        # global _current_model, _current_cfg
        # _current_model = model
        # _current_cfg = cfg
        #
        # for epoch in range(cfg.training.epochs):
        #     model.train()
        #     for step, batch in enumerate(train_loader):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         output = model(batch["input"])
        #         loss = compute_loss(output, batch["label"])
        #         loss.backward()
        #         optimizer.step()
        #         optimizer.zero_grad()
        #         log_metrics({"train/loss": loss.item()}, step=step)
        #
        #         if step % cfg.training.eval_every == 0:
        #             model.eval()
        #             eval_metrics = run_eval(model, val_loader, device)
        #             log_metrics(eval_metrics, step=step)

        logger.info("Training loop placeholder — implement your domain-specific logic here")

        # Placeholder: log a dummy metric so the run has data
        log_metrics({"placeholder/loss": 0.5}, step=0)

        # 7. Finish W&B run
        finish_run(run, status="success")

    except Exception:
        logger.exception("Training failed")
        finish_run(run, status="failed")
        raise

    finally:
        # 8. Sync results to storage
        sync_results()
        logger.info("Results synced to storage")


if __name__ == "__main__":
    main()
