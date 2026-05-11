# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "wandb", "pandas", "matplotlib"]
# ///

"""W&B Experiment Dashboard

This marimo notebook provides an interactive dashboard for browsing and
analyzing experiment runs from Weights & Biases, cross-referenced with
the experiment registry.

Usage:
    uv run marimo edit notebooks/dashboard.py
    uv run marimo run notebooks/dashboard.py --no-show
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def _():
    """Import required libraries."""
    import os
    from datetime import datetime

    import matplotlib.pyplot as plt
    import pandas as pd
    import wandb

    # Configure W&B project from environment
    WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "[[ package_name ]]")
    WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

    return WANDB_ENTITY, WANDB_PROJECT, datetime, os, pd, plt, wandb


@app.cell
@mo.cache(pin=True)
def _fetch_runs(WANDB_PROJECT, WANDB_ENTITY, wandb):
    """Fetch all runs from W&B.

    This cell is marked as a data-fetching cell. It calls the W&B API
    and caches results for performance.
    """
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT)

    runs_data = []
    for run in runs:
        runs_data.append(
            {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,
                "tags": run.tags,
                **run.config,
                **{
                    f"summary/{k}": v for k, v in run.summary.items() if isinstance(v, (int, float))
                },
            }
        )

    return pd.DataFrame(runs_data) if runs_data else pd.DataFrame()


@app.cell
@mo.cache(pin=True)
def _load_registry(os, pd):
    """Load experiment registry for cross-referencing.

    Data-fetching cell: reads the local registry YAML.
    """
    registry_path = "experiments/registry.yaml"
    if os.path.exists(registry_path):
        import yaml

        with open(registry_path) as f:
            data = yaml.safe_load(f)
        experiments = data.get("experiments", [])
        return pd.DataFrame(experiments) if experiments else pd.DataFrame()
    return pd.DataFrame()


@app.cell
def _filter_ui(mo):
    """Create filter UI widgets."""
    search = mo.ui.text(label="Search runs", placeholder="Filter by name or tag...")
    state_filter = mo.ui.dropdown(
        options=["all", "running", "finished", "crashed", "failed"],
        value="all",
        label="State",
    )
    return mo.hstack([search, state_filter]), search, state_filter


@app.cell
def _filtered_table(df_runs, search, state_filter):
    """Display filtered runs table."""
    if df_runs.empty:
        return mo.md("*No runs found in W&B project.*")

    filtered = df_runs.copy()

    # Apply state filter
    if state_filter.value != "all":
        filtered = filtered[filtered["state"] == state_filter.value]

    # Apply search filter
    if search.value:
        query = search.value.lower()
        mask = filtered["name"].str.lower().str.contains(query, na=False) | filtered["tags"].apply(
            lambda t: any(query in str(tag).lower() for tag in (t or []))
        )
        filtered = filtered[mask]

    return mo.ui.table(filtered, selection=None)


@app.cell
@mo.cache(pin=True)
def _fetch_metrics(selected_run_id, WANDB_PROJECT, WANDB_ENTITY, wandb):
    """Fetch metrics history for a selected run.

    Data-fetching cell: queries W&B for run history.
    """
    if not selected_run_id:
        return pd.DataFrame()

    api = wandb.Api()
    run_path = (
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/{selected_run_id}"
        if WANDB_ENTITY
        else f"{WANDB_PROJECT}/{selected_run_id}"
    )
    run = api.run(run_path)
    history = run.history(pandas=True)
    return history if not history.empty else pd.DataFrame()


@app.cell
def _plot_metrics(df_history):
    """Plot training curves for selected runs."""
    if df_history.empty:
        return mo.md("*Select a run to view training curves.*")

    numeric_cols = [c for c in df_history.columns if df_history[c].dtype in ("float64", "int64")]
    if not numeric_cols:
        return mo.md("*No numeric metrics available for plotting.*")

    fig, axes = plt.subplots(
        nrows=min(len(numeric_cols), 4),
        ncols=1,
        figsize=(10, 3 * min(len(numeric_cols), 4)),
    )
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols[:4]):
        ax.plot(df_history[col], linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    app.run()
