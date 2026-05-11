# Claude Code Instructions

Project: `[[ package_name ]]` - ML Research Project Template

## Common Commands

| Command | Description |
|---------|-------------|
| `just install` | Install dependencies via uv sync |
| `just check` | Run all checks (lint + typecheck + test) |
| `just lint` | Format and lint with ruff |
| `just typecheck` | Type check with pyright |
| `just test` | Run pytest suite |
| `just train` | Run training with default experiment |
| `just train experiment=base` | Run specific experiment |
| `just sweep` | Launch wandb hyperparameter sweep |
| `just figures` | Generate all marimo figures |
| `just status` | Show running jobs and recent entries |
| `just dashboard` | Open wandb dashboard |
| `just clean` | Remove build artifacts |

## Architecture Overview

```
src/[[ package_name ]]/
├── __init__.py           # Package init
├── train.py              # CLI entry point: loads config, runs training loop
├── models.py             # Model definitions (torch.nn.Module subclasses)
├── data.py               # Dataset classes and DataLoader factories
├── config.py             # Pydantic v2 settings (TrainingConfig, ModelConfig, DataConfig)
├── infra/
│   ├── storage.py        # fsspec wrapper: sync to/from S3, GCS, local
│   ├── registry.py       # YAML-based experiment registry
│   └── journal.py        # Run tracking: status files in journal/runs/
└── analysis/
    ├── results.py        # Load run configs, metrics, checkpoints
    └── plotting.py       # Marimo notebook helpers for figures
```

### Training Flow

1. `train.py` parses CLI args (`--experiment`, `--config`, `--run-id`)
2. Loads config from YAML via Pydantic (in `config.py`)
3. Sets up wandb logging
4. Instantiates model (`models.py`) and data loaders (`data.py`)
5. Runs training loop with checkpoint saving
6. On completion: pushes results via `storage.py`, updates registry

### Config System

Configs are hierarchical Pydantic v2 models:

```python
class TrainingConfig(BaseSettings):
    """Training configuration validated by Pydantic."""
    model: ModelConfig
    train: OptimizerConfig
    data: DataConfig
    logging: LoggingConfig
```

Load from YAML: `config = TrainingConfig.from_yaml("experiments/base/config.yaml")`

Environment variables override YAML values via `pydantic-settings`.

## Code Style Conventions

- **Python 3.11+** with `from __future__ import annotations`
- **Strict typing**: All functions annotated, no `Any` without justification
- **Double quotes** for strings
- **100 character** line limit
- **Loguru** for logging: `from loguru import logger; logger.info("...")`
- **Rich** for CLI output: `from rich.console import Console; console = Console()`
- **fsspec** for all I/O: `from fsspec import filesystem`
- **Pydantic** for all config/data validation
- Prefer `pathlib.Path` over string paths
- Use `tqdm` (via Rich) for progress bars
- Context managers for resource management

### Import Order

```python
from __future__ import annotations

# 1. stdlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

# 2. third-party
import torch
import yaml
from pydantic import BaseModel, Field
from loguru import logger

# 3. local
from [[ package_name ]].config import TrainingConfig
from [[ package_name ]].models import MyModel
```

## Project-Specific Patterns

### Experiment Registry

All runs are tracked in `experiments/registry.yaml`. Each entry records:
- `run_id`: Unique identifier
- `experiment`: Experiment config name
- `status`: pending / running / completed / failed
- `start_time`, `end_time`: ISO timestamps
- `config_path`: Path to full config YAML

### Storage Backend

The `STORAGE_BACKEND` env var controls where results go:
- `local` -> `STORAGE_ROOT` (default: `./results`)
- `s3` -> `s3://{S3_BUCKET}/{run_id}/`
- `gcs` -> `gs://{GCS_BUCKET}/{run_id}/`

Use `infra.storage` module, never direct boto3 / google.cloud calls.

### Journal Files

Each run writes a status YAML to `journal/runs/{run_id}.yaml`. Launchers update these throughout the run lifecycle.

### Adding a New Eval Script

1. Create `evals/{eval_name}.py`
2. Accept `--run-id` argparse argument
3. Load model via `analysis.results.load_checkpoint(run_id)`
4. Save results as JSON to `results/{run_id}/{eval_name}.json`
5. Add to `evals/` for `just eval-all` to pick it up automatically
