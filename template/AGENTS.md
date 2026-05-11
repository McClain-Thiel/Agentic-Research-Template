# AI Agent Instructions

This document provides guidance for AI agents working on the `[[ package_name ]]` project.

> For detailed Claude Code specific instructions, see [CLAUDE.md](CLAUDE.md).

## Project Conventions

- **Python 3.11+** with strict type annotations throughout
- **Pydantic** models for all configuration and data validation
- **fsspec** for all filesystem operations (local, S3, GCS)
- **Loguru** for logging, **Rich** for pretty printing
- **wandb** for experiment tracking
- **Hugging Face Hub** for model and dataset artifacts

## Before Making Changes

1. Read the relevant module docstrings and type signatures
2. Check `experiments/` for config examples
3. Run `just check` after any code changes
4. Add tests for new functionality in `tests/`

## Code Style

- Follow PEP 8 conventions enforced by ruff
- Use double quotes for strings
- Maximum line length: 100 characters
- Import ordering: stdlib, third-party, local (enforced by ruff)
- All public functions must have type annotations and docstrings
- Use `pathlib.Path` for filesystem paths
- Prefer composition over inheritance

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/[[ package_name ]]/config.py` | Pydantic configuration models |
| `src/[[ package_name ]]/train.py` | Main training entry point |
| `src/[[ package_name ]]/models.py` | Model architecture definitions |
| `src/[[ package_name ]]/data.py` | Dataset loading and preprocessing |
| `src/[[ package_name ]]/infra/storage.py` | fsspec-based storage operations |
| `src/[[ package_name ]]/infra/registry.py` | Experiment registry management |
| `src/[[ package_name ]]/analysis/results.py` | Run analysis utilities |

## Testing

- All tests go in `tests/` mirroring the `src/` structure
- Run `just test` to execute the test suite
- Tests should not require network access unless explicitly marked
