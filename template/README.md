# [[ project_name ]]

[[ description ]]

Generated from [ml-research-template](https://github.com/mcclain-thiel/ml-research-template).

## Quick start

Prerequisites: [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                  # install dependencies
cp .env.example .env     # fill in WANDB_API_KEY, HF_TOKEN, etc.
just check               # lint + typecheck + test
just train               # placeholder training run
```

## Layout

```
src/[[ package_name ]]/
├── train.py          # CLI entrypoint + training loop (placeholder)
├── config/           # Pydantic settings + experiment configs
├── data/             # Dataset + DataLoader stubs
├── models/           # BaseModel with save/load
├── infra/            # storage (fsspec), logging (W&B), reproducibility
└── analysis/         # results, metrics, registry access

experiments/          # YAML experiment configs + registry.yaml
tests/                # config / shapes / data / reproducibility
launchers/            # local.sh (functional); others are stubs
figures/, notebooks/  # marimo
```

## Common tasks

| Command | What it does |
|---|---|
| `just install` | `uv sync` |
| `just check` | lint + typecheck + test |
| `just train` | run `experiments/base/config.yaml` via `LAUNCHER` (default `local`) |
| `just train experiment=foo` | run `experiments/foo/config.yaml` |
| `just sweep` | launch a W&B sweep from `configs/sweep.yaml` |
| `just figures` | execute all marimo figure notebooks headlessly |
| `just status` | show running jobs and recent registry entries |
| `just dashboard` | open the W&B project URL |
| `just push-results` / `just pull-results` | sync `results/` to/from remote storage |
| `just reproduce <run_id>` | print the command to reproduce a past run |

## Configs

Experiments live in `experiments/<name>/config.yaml` and validate against `[[ package_name ]].config.experiment.ExperimentConfig`. Override at the CLI:

```bash
uv run python -m [[ package_name ]].train \
    --experiment experiments/base \
    --config training.lr=3e-4 \
    --config model.hidden_size=512
```

## Storage

Set `STORAGE_BACKEND` in `.env` to `local`, `s3`, or `gcs`. All I/O goes through `[[ package_name ]].infra.storage` — no direct `boto3`/`google.cloud` calls.

## Updating from the template

This repo was generated from a copier template. To pull in upstream improvements:

```bash
uvx copier update
```

Your answers are stored in `.copier-answers.yml`. You'll get a 3-way diff for anything you've edited.

## License

[Add your license here]
