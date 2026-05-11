"""Shared matplotlib style for [[ package_name ]] figures.

Provides consistent styling across all publication figures.
Import and call `apply_style()` before creating plots.

Example:
    >>> from figures.style import apply_style
    >>> import matplotlib.pyplot as plt
    >>> apply_style()
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> fig.savefig("figures/output/my_fig.pdf")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

# Color palette -- muted, publication-friendly
COLORS = [
    "#4C72B0",  # Blue
    "#DD8452",  # Orange
    "#55A868",  # Green
    "#C44E52",  # Red
    "#8172B3",  # Purple
    "#937860",  # Brown
    "#DA8BC3",  # Pink
    "#8C8C8C",  # Gray
    "#CCB974",  # Yellow
    "#64B5CD",  # Cyan
]

# Line styles for multi-line plots
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Marker styles
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def apply_style(
    figsize: tuple[float, float] = (3.5, 2.5),
    fontsize: int = 9,
    linewidth: float = 1.2,
    dpi: int = 300,
) -> None:
    """Apply the project's shared matplotlib style.

    Configures figure defaults, font sizes, color palette, and other
    style parameters for consistent publication-quality figures.

    Args:
        figsize: Default figure size in inches (width, height).
        fontsize: Base font size for all text elements.
        linewidth: Default line width for plots.
        dpi: Default resolution for saved figures.
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "figure.constrained_layout.use": True,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Font
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize + 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "figure.titlesize": fontsize + 2,
            # Lines
            "lines.linewidth": linewidth,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Colors
            "axes.prop_cycle": cycler(color=COLORS),
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            # Backend
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# Convenience re-exports
__all__ = ["apply_style", "COLORS", "LINESTYLES", "MARKERS"]
