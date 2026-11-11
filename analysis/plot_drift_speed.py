"""Generate fig/drift_speed.pdf: how the distributions mix at each drift speed.

Run with `uv run analysis/plot_drift_speed.py` from the repository root.
The stream is RFMNIST: 5 equal tasks. Each drift follows the sigmoid of
`capymoa.ocl.datasets.gradual.SigmoidFn`. The x-axis is progress in %, the y-axis the
share of each distribution. Vertical lines mark the drift centres.
"""

import matplotlib.pyplot as plt
import numpy as np
from common import (
    FONT_LABEL,
    FONT_SUBTITLE,
    FONT_TICK,
    NO_TOP_RIGHT,
    PAPER_STYLE,
)
from plots import save_fig

# First two are fixed, the rest are Okabe-Ito
COLORS = ("#5D3C99", "#1E9E89", "#E69F00", "#D55E00", "#CC79A7")
N_TASKS = len(COLORS)
# Drift speeds as (label, width), same widths as config/boundary
SPEEDS = (("abrupt", 0.0), ("gradual", 0.1), ("slow", 0.2))
N_POINTS = 801
# Equal tasks: drift k sits at k / N_TASKS, the sigmoid scale is w * task length
TASK_PCT = 100 / N_TASKS
DRIFT_CENTERS = TASK_PCT * np.arange(1, N_TASKS)


def drift_share(progress: np.ndarray, center: float, width: float) -> np.ndarray:
    """Share of the later distribution at each progress value (in %) around a drift."""
    if width == 0.0:
        return (progress >= center).astype(float)
    return 1 / (1 + np.exp(-4 / (width * TASK_PCT) * (progress - center)))


def task_shares(progress: np.ndarray, width: float) -> np.ndarray:
    """Share of each distribution, shape (N_TASKS, len(progress))."""
    # after[k] is the share of tasks later than k, so task k = after[k-1] - after[k]
    after = [np.ones_like(progress)]
    after += [drift_share(progress, c, width) for c in DRIFT_CENTERS]
    after.append(np.zeros_like(progress))
    return np.array([after[k] - after[k + 1] for k in range(N_TASKS)])


def main() -> None:
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(NO_TOP_RIGHT)
    progress = np.linspace(0, 100, N_POINTS)
    fig, axes = plt.subplots(
        len(SPEEDS),
        1,
        figsize=(3.3, 1.95),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.03)
    for ax, (label, width) in zip(axes, SPEEDS):
        ax.stackplot(
            progress, task_shares(progress, width), colors=COLORS, linewidth=0
        )
        for x in DRIFT_CENTERS:
            ax.axvline(x, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(
            f"{label.capitalize()} (w={width})", fontsize=FONT_SUBTITLE, pad=2
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.minorticks_off()
        ax.tick_params(labelsize=FONT_TICK)
    axes[-1].set_xlabel(r"Progress (\%)", fontsize=FONT_LABEL)
    fig.supylabel("Proportion", fontsize=FONT_LABEL)
    save_fig(fig, "drift_speed.pdf")


if __name__ == "__main__":
    main()
