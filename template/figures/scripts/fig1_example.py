# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "matplotlib", "pandas"]
# ///

"""Figure 1: Example Figure

This notebook demonstrates the standard pattern for creating publication
figures in this project:
1. Load evaluation results from the results/ directory
2. Apply the shared style from figures.style
3. Create the figure with matplotlib
4. Save to figures/output/

Usage:
    # Edit interactively:
    uv run marimo edit figures/scripts/fig1_example.py

    # Run headless (for the `just figures` target):
    uv run marimo run figures/scripts/fig1_example.py --no-show

    # Export as PDF:
    uv run marimo export pdf figures/scripts/fig1_example.py -o figures/output/fig1_example.pdf
"""

from __future__ import annotations

from pathlib import Path

import marimo

__generated_with = "0.8.0"
app = marimo.App()


@app.cell
def _():
    """Import libraries and configure paths."""
    import json

    import matplotlib.pyplot as plt
    import pandas as pd

    from figures.style import apply_style

    # Apply shared style
    apply_style(figsize=(3.5, 2.5))

    # Configure paths
    RESULTS_DIR = Path("results")
    OUTPUT_DIR = Path("figures/output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return OUTPUT_DIR, RESULTS_DIR, apply_style, json, plt, pd


@app.cell
@mo.cache(pin=True)
def _load_results(RESULTS_DIR, pd, json):
    """Load evaluation results for plotting.

    TODO: Replace with your actual result loading logic.
    This loads all .json files from results/ subdirectories.
    """
    results = []
    if RESULTS_DIR.exists():
        for run_dir in RESULTS_DIR.iterdir():
            if run_dir.is_dir():
                for result_file in run_dir.glob("*.json"):
                    try:
                        with open(result_file) as f:
                            data = json.load(f)
                        data["run_id"] = run_dir.name
                        data["eval_file"] = result_file.name
                        results.append(data)
                    except (json.JSONDecodeError, OSError):
                        continue

    if not results:
        # Generate placeholder data for demonstration
        results = [
            {"run_id": "baseline", "epoch": 1, "accuracy": 0.65, "loss": 0.85},
            {"run_id": "baseline", "epoch": 2, "accuracy": 0.72, "loss": 0.62},
            {"run_id": "baseline", "epoch": 3, "accuracy": 0.78, "loss": 0.48},
            {"run_id": "baseline", "epoch": 4, "accuracy": 0.82, "loss": 0.38},
            {"run_id": "baseline", "epoch": 5, "accuracy": 0.85, "loss": 0.32},
            {"run_id": "improved", "epoch": 1, "accuracy": 0.70, "loss": 0.80},
            {"run_id": "improved", "epoch": 2, "accuracy": 0.79, "loss": 0.55},
            {"run_id": "improved", "epoch": 3, "accuracy": 0.86, "loss": 0.40},
            {"run_id": "improved", "epoch": 4, "accuracy": 0.90, "loss": 0.30},
            {"run_id": "improved", "epoch": 5, "accuracy": 0.92, "loss": 0.25},
        ]

    return pd.DataFrame(results)


@app.cell
def _create_figure(df, plt, OUTPUT_DIR):
    """Create and save the figure."""
    if df.empty:
        fig = plt.figure(figsize=(3.5, 2.5))
        plt.text(0.5, 0.5, "No results found", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Plot each run's accuracy curve
    if "run_id" in df.columns and "epoch" in df.columns:
        for run_id, group in df.groupby("run_id"):
            ax.plot(
                group["epoch"],
                group["accuracy"],
                marker="o",
                markersize=3,
                label=run_id,
                linewidth=1.2,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training Accuracy")
        ax.legend(frameon=True, loc="lower right")
    else:
        # Fallback: simple bar plot
        df.head(10).plot(kind="bar", ax=ax)
        ax.set_title("Results Summary")

    # Save figure
    output_path = OUTPUT_DIR / "fig1_example.pdf"
    fig.savefig(output_path, format="pdf")
    print(f"Figure saved to {output_path}")

    return fig


if __name__ == "__main__":
    app.run()
