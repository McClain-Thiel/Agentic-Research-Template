# Agentic ML Research

A [copier](https://copier.readthedocs.io/) template for ML research projects. Run it via `uvx` and get a working repo with sane defaults: typed config, experiment tracking, storage abstraction, tests, and a task runner — without the ceremony of cookiecutter clones or one-off setup scripts.

## Use it

```bash
uvx copier copy gh:mcclain-thiel/Agentic-Research-Template ./my-new-project --trust
```

That's it. Copier prompts for a handful of variables (project name, package name, author, Python version, storage backend, W&B entity), writes the files, runs `uv sync`, initializes a git repo, and makes an initial commit. From there:

```bash
cd my-new-project
just check     # lint + typecheck + test
just train     # runs the placeholder training loop
```

To pick up template improvements later in an existing generated project:

```bash
uvx copier update
```

Copier remembers your answers in `.copier-answers.yml` and re-applies the template — you get a 3-way diff for anything you've edited.

## Philosophy

The template encodes opinions about what an ML research project should look like on day one. Each choice is deliberate and reversible.

- **Configs are typed and validated.** Experiments are Pydantic v2 models loaded from YAML. Invalid configs fail loudly at load time, not 40 minutes into a training run. CLI overrides (`--config training.lr=3e-4`) get the same validation as YAML values.
- **The training loop is yours; the scaffolding isn't.** `train.py` is small — argument parsing, seed-setting, W&B init, results sync. The actual loop is a commented-out placeholder. The point is the surrounding plumbing, not opinions about your model or optimizer.
- **One I/O abstraction: fsspec.** Switching between local disk, S3, and GCS is a config change, not a code change. No direct `boto3` / `google.cloud` calls in the project. `STORAGE_BACKEND=s3` in `.env` is the entire migration.
- **Experiment tracking is non-optional.** Every run logs to W&B with hyperparameters, system metrics, git hash, and a copy of the config. If you don't want W&B, swap the four functions in `infra/logging.py` — but the surface area is small enough that you have to make a deliberate choice.
- **Reproducibility is a default, not a feature.** Seeds are set, git status is captured, environment is logged, and runs land in a registry (`experiments/registry.yaml`) plus a per-run journal file. You can hand a `run_id` to a collaborator and they can reconstruct the run.
- **One launcher, room for more.** `local.sh` works out of the box. `slurm.sh`, `ec2.sh`, `aws_batch.py`, and `ray.py` ship as stubs — they fail loudly with "implement me" rather than pretending to work. Add them when you actually have the compute, not before.
- **Strict typing, strict lint.** `pyright --strict` and `ruff` are wired into `just check`. Pre-commit hooks catch them before they reach CI. The bar is high so new code doesn't drift.
- **The harness lives next to the code.** `CLAUDE.md`, `AGENTS.md`, and `.claude/` are checked in so an AI assistant working in the repo has accurate context. Treating agent instructions as code, not folklore.

## What you get

```
my-new-project/
├── src/my_package/
│   ├── train.py              # CLI + training loop (mostly empty)
│   ├── config/               # Pydantic settings + experiment configs
│   ├── data/                 # Dataset and DataLoader stubs
│   ├── models/               # Base model with save/load
│   ├── infra/                # Storage (fsspec), logging (W&B), reproducibility
│   └── analysis/             # Result loading, metrics, registry access
├── tests/                    # Config round-trip, shapes, dataloader, seed determinism
├── experiments/              # YAML configs + registry.yaml
├── launchers/                # local.sh (functional) + stubs
├── figures/                  # Marimo figure notebooks
├── notebooks/                # Marimo W&B dashboard
├── justfile                  # install / lint / typecheck / test / train / sweep
├── pyproject.toml            # uv-managed, hatchling build
├── Dockerfile                # Multi-stage: base / train / dev
├── .devcontainer/            # VS Code dev container
├── .pre-commit-config.yaml   # ruff + nbstripout + detect-secrets
├── CLAUDE.md / AGENTS.md     # Agent-facing project docs
└── .env.example              # All secrets/config in one place
```

## Prompts

| Variable | Purpose | Default |
|---|---|---|
| `project_name` | Display name | `ML Research Project` |
| `package_name` | Python identifier (importable) | derived from `project_name` |
| `project_slug` | Dir / PyPI-style name | derived from `package_name` |
| `description` | One-line description for `pyproject.toml` | placeholder |
| `author_name`, `author_email` | `pyproject.toml` authors field | placeholder |
| `python_version` | Min Python (3.11 / 3.12 / 3.13) | `3.11` |
| `storage_backend` | `local` / `s3` / `gcs` — pulls in `s3fs` or `gcsfs` automatically | `local` |
| `wandb_entity` | Default W&B team/user | empty |

## Requirements

- [`uv`](https://docs.astral.sh/uv/) — `uvx` ships with it.
- `git` for the post-generation init.
- Python 3.11+ available to `uv` (it'll fetch one if needed).

## Why copier, not cookiecutter

Copier supports `copier update` — when the template gets better, generated projects can pull the changes in. Cookiecutter is a one-shot; there's no relationship after generation. For a template you actually plan to maintain, that difference matters.

## License

MIT. Generated projects choose their own license.
