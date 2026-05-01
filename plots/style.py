"""Shared matplotlib styling constants."""
import numpy as np
import matplotlib.pyplot as plt

BASELINE_BPB = 1.36785
OUTLIER_THRESHOLD = 2.0
PHASE_CMAP = plt.cm.tab10


def phase_colours(phases):
    """Map phase names to colours."""
    phases = sorted(set(phases))
    return dict(
        zip(phases, PHASE_CMAP(np.linspace(0, 0.9, len(phases))))
    )


def edge_colour(valid):
    v = valid.lower()
    if v == "yes":
        return "#2ca02c"
    if v == "no":
        return "#d62728"
    return "#ff7f0e"


def delta_colour(delta_val):
    if delta_val < 0:
        return "#2ca02c"   # green = improved
    if delta_val > 0:
        return "#d62728"   # red = worsened
    return "#888888"


def apply_base_spines(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#555555")
        ax.spines[sp].set_linewidth(1.2)


def setup_figure(figsize=(15, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    return fig, ax
