import os
# Run with `uv run notebook/plot_error_stream.py`
import pickle
from typing import Sequence
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import withStroke
import numpy as np
from pathlib import Path
from common import (
    ONE_SIDED_LABELS,
    DRIFT_WINDOW_COLOR,
    FONT_LABEL,
    FONT_SUBTITLE,
    FONT_TICK,
    FONT_TITLE,
    NO_TOP_RIGHT,
    PRED_DRIFT_COLOR,
    TRUE_DRIFT_COLOR,
)
import scienceplots as _  # noqa: F401

plt.style.use(["science", "nature"])
plt.rcParams.update(NO_TOP_RIGHT)

# %%
METRIC_FONTSIZE = FONT_TICK
# Zero sits on the bottom spine. The chevrons of the predicted drifts stand just above it,
# clear of the axis ticks.
Y_MIN = 0
PRED_CHEVRON_Y = 0.1
# A "V" with its tip down. The arms sit on the marker position, the tip below it.
CHEVRON = MplPath(
    [(-1, 0), (0, -1), (1, 0)], [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO]
)
# The same chevron turned over: a "^" with its tip up, standing on the marker position.
CHEVRON_UP = MplPath(
    [(-1, 0), (0, 1), (1, 0)], [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO]
)
# Accuracy on seen tasks is the focus, so it is black. The error rate is Okabe-Ito blue.
ACCURACY_SEEN_COLOR = "black"
ACCURACY_SEEN_TEXT_COLOR = "black"
ERROR_RATE_COLOR = "#0072B2"
# Narrower than a column: the saved PDF adds padding and must stay within 252 pt
FIGSIZE_ERROR_STREAM = (248 / 72, 3.1)


def plot_error_stream(metrics, ax: plt.Axes, ocl_metrics, summary: bool = True):
    accuracy_seen_avg: float = ocl_metrics.accuracy_seen_avg
    backward_transfer: float = ocl_metrics.backward_transfer

    trues = np.array(metrics["trues"])
    preds = np.array(metrics["preds"])
    error_stream: Sequence[bool] = metrics["error_stream"]
    max_early: int = metrics["max_early_detection"]
    max_delay: int = metrics["max_delay"]
    f1: float = metrics["my_f1"]
    tp: int = metrics["my_tp"]
    fp: int = metrics["my_fp"]
    fn: int = metrics["my_fn"]

    n_samples = 512
    window_size = 512
    windowed_error_stream = np.convolve(
        error_stream, np.ones(window_size) / window_size, mode="valid"
    )
    xs = (
        np.linspace(0, len(windowed_error_stream), n_samples, endpoint=False)
        .round()
        .astype(int)
    )
    ys = windowed_error_stream[xs]

    # Show positions as a percentage of the stream
    to_pct = 100 / len(error_stream)
    xs = xs * to_pct
    trues = trues * to_pct
    preds = preds * to_pct
    max_early = max_early * to_pct
    max_delay = max_delay * to_pct

    ax.plot(xs, ys, color=ERROR_RATE_COLOR, linewidth=0.6)

    # Accuracy on all tasks seen so far, measured at the end of each task
    task_ends = np.asarray(ocl_metrics.boundaries)[1:] / ocl_metrics.boundaries[-1] * 100
    ax.plot(
        task_ends, ocl_metrics.accuracy_seen, color=ACCURACY_SEEN_COLOR, linewidth=0.6,
        marker="s", markersize=2.5, markeredgewidth=0, zorder=4, clip_on=False,
    )
    for i, (x, v) in enumerate(zip(task_ends, ocl_metrics.accuracy_seen)):
        # The markers sit on drift lines, so put the numbers to the right of the line.
        # The last one has no line, so it is centred.
        last = i == len(task_ends) - 1
        ax.annotate(
            f"{100 * v:.0f}", xy=(x, v), xytext=(0 if last else 3, 2.5),
            textcoords="offset points",
            ha="center" if last else "left", va="bottom", fontsize=FONT_TICK,
            color=ACCURACY_SEEN_TEXT_COLOR,
            zorder=5, path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )

    # True drifts: a chevron hanging from the top spine, tip pointing down
    ax.plot(
        trues, np.ones(len(trues)), linestyle="none", marker=CHEVRON, color=TRUE_DRIFT_COLOR,
        markersize=5, markeredgewidth=1.0, fillstyle="none", clip_on=False, zorder=5,
    )
    # Predicted drifts: a chevron standing just above the bottom spine, tip pointing up.
    # They are clipped to the axes (the layout breaks if they stick out).
    ax.set_ylim(Y_MIN, 1.05)
    ax.plot(
        preds, np.full(len(preds), PRED_CHEVRON_Y), linestyle="none", marker=CHEVRON_UP,
        color=PRED_DRIFT_COLOR, markersize=5, markeredgewidth=1.0, fillstyle="none",
        zorder=5,
    )
    # Fill between the trues - max_early and trues + max_delay
    for true in trues:
        ax.fill_betweenx(
            [0, 1], true - max_early, true + max_delay, color=DRIFT_WINDOW_COLOR, alpha=1
        )

    # The metrics go in the title row, so they hide none of the stream
    if summary:
        ax.set_title(
            rf"Acc. {accuracy_seen_avg * 100:.1f}\%~~F1 {f1 * 100:.1f}\%~~BWT ${backward_transfer:.2f}$"
            rf"~~TP/FP/FN {tp}/{fp}/{fn}",
            loc="right", fontsize=METRIC_FONTSIZE,
        )


def load_and_plot_error_stream(
    metrics_path: Path | str, ax: plt.Axes, summary: bool = True
):
    metrics_path = Path(metrics_path)
    dd_metrics = pickle.load(open(metrics_path / "dd_metrics.pkl", "rb"))
    ocl_metrics = pickle.load(open(metrics_path / "ocl_metrics.pkl", "rb"))
    plot_error_stream(dd_metrics, ax, ocl_metrics, summary=summary)


BOUNDARIES = ("abrupt", "gradual", "slow")

# Log folder of each detector name, where they differ
RUN_DIR = {name: run for run, name in ONE_SIDED_LABELS.items()}

# Two columns wide (7.16 in text width), a little under so the saved PDF fits
FIGSIZE_TWO_COLUMN = (511 / 72, 2.7)


def add_legend(fig: plt.Figure, ncol: int) -> None:
    """One legend under the whole figure, for the lines, the windows and the accuracy."""
    fig.legend(
        [
            Line2D([], [], color=ERROR_RATE_COLOR, linewidth=0.6),
            Line2D(
                [], [], color=ACCURACY_SEEN_COLOR, linewidth=0.6, marker="s",
                markersize=2.5, markeredgewidth=0,
            ),
            Line2D(
                [], [], linestyle="none", marker=CHEVRON, color=TRUE_DRIFT_COLOR,
                markersize=5, markeredgewidth=1.0, fillstyle="none",
            ),
            Line2D(
                [], [], linestyle="none", marker=CHEVRON_UP, color=PRED_DRIFT_COLOR,
                markersize=5, markeredgewidth=1.0, fillstyle="none",
            ),
            Patch(facecolor=DRIFT_WINDOW_COLOR),
        ],
        ["Error rate", "Accuracy on seen tasks", "True drift", "Predicted drift",
         "Detection window"],
        loc="outside lower center", ncol=ncol, frameon=False, fontsize=FONT_LABEL,
        handlelength=1.2, columnspacing=1.0, handletextpad=0.4,
    )


def plot_drift_detection_pair(
    root: Path,
    good: tuple[str, str] = ("LWF", "ADWIN"),
    bad: tuple[str, str] = ("MAS", "STEPD"),
    seed: str = "000",
    names: tuple[str, str] = ("High F1", "Low F1"),
) -> plt.Figure:
    """Error streams for a left and a right run, one row per boundary.

    `good` and `bad` are (strategy, detector) pairs. `names` start the column headings.
    """
    fig = plt.figure(figsize=FIGSIZE_TWO_COLUMN, layout="constrained")
    subfigs = fig.subfigures(1, 2)
    for col, (sf, name, (strategy, detector)) in enumerate(
        zip(subfigs, names, (good, bad))
    ):
        axs = sf.subplots(nrows=3, sharex=True, sharey=True)
        for row, (ax, boundary) in enumerate(zip(axs, BOUNDARIES)):
            load_and_plot_error_stream(
                root / f"evaluate/{boundary}/{strategy}/{RUN_DIR.get(detector, detector)}/{seed}/", ax
            )
            ax.set_title(boundary.capitalize(), loc="left", fontsize=FONT_SUBTITLE)
            ax.minorticks_off()
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=FONT_TICK)
        sf.suptitle(
            f"{name}: {strategy} with {detector}", fontsize=FONT_TITLE
        )
        axs[2].set_xlabel(r"Stream (\%)", fontsize=FONT_LABEL)
        if col == 0:
            axs[1].set_ylabel("Rate", fontsize=FONT_LABEL)

    add_legend(fig, ncol=5)
    return fig


fig = plot_drift_detection_pair(
    Path(os.environ.get("LOGDIR", "logs")) / "cldd",
    good=("LWF", "ADWIN"),
    bad=("LWF", "STEPD"),
    seed="005",
)
fig.savefig("fig/plot_error_stream.pdf")

# A detector that misses drifts: no alarm at all in Gradual and Slow
fig_missed = plot_drift_detection_pair(
    Path(os.environ.get("LOGDIR", "logs")) / "cldd",
    good=("LWF", "ADWIN"),
    bad=("LWF", "DDM"),
    seed="005",
    names=("Finds drifts", "Misses drifts"),
)
fig_missed.savefig("fig/plot_error_stream_missed.pdf")

# One column wide (3.5 in), a little under so the saved PDF fits
FIGSIZE_ONE_COLUMN_STACK = (248 / 72, 4.4)


# Letters the caption refers to: (row, letter, stream %, rate), rows counted from the top
STACK_LETTERS = [
    (1, "a", 70, 0.95),  # ADWIN: accuracy falls slowly
    (3, "b", 55, 0.4),  # STEPD: false alarms outside the windows
    (4, "c", 50, 0.3),  # DDM: no alarm at all
    (4, "d", 90, 0.62),  # DDM: accuracy falls although the error stays low
]


def plot_drift_detection_stack(
    root: Path,
    runs: Sequence[tuple[str, str, str, Sequence[str]]],
    seed: str = "000",
) -> plt.Figure:
    """Error streams in one column: a group per run, one row per boundary of that run.

    `runs` holds (heading start, strategy, detector, boundaries) tuples, top to bottom.
    All rows are the same height. Each group has its heading above its first row.
    """
    n_rows = sum(len(boundaries) for *_, boundaries in runs)
    fig, axs = plt.subplots(
        n_rows, 1, figsize=FIGSIZE_ONE_COLUMN_STACK, layout="constrained",
        sharex=True, sharey=True,
    )
    row = 0
    for g, (name, strategy, detector, boundaries) in enumerate(runs):
        for k, boundary in enumerate(boundaries):
            ax = axs[row]
            row += 1
            load_and_plot_error_stream(
                root / f"evaluate/{boundary}/{strategy}/{RUN_DIR.get(detector, detector)}/{seed}/", ax, summary=False
            )
            # The boundary is a label inside the plot, at the top left by the y axis
            ax.text(
                0.055, 0.97, boundary.capitalize(), transform=ax.transAxes,
                ha="left", va="top", fontsize=FONT_LABEL, zorder=6,
            )
            ax.minorticks_off()
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=FONT_TICK, labelbottom=(k == len(boundaries) - 1))
            if k == 0:
                ax.set_title(f"{name}: {strategy} with {detector}", fontsize=FONT_TITLE)
    # One x label, under the last row
    axs[-1].set_xlabel(r"Stream (\%)", fontsize=FONT_LABEL)
    # Red letters mark the regions the caption talks about
    for row, letter, x, y in STACK_LETTERS:
        axs[row].text(
            x, y, rf"\textbf{{{letter}}}", color="#8B0000", fontsize=FONT_SUBTITLE,
            ha="center", va="center", zorder=7,
            path_effects=[withStroke(linewidth=2.2, foreground="white")],
        )

    # One y label for all rows
    fig.supylabel("Rate", fontsize=FONT_LABEL)
    add_legend(fig, ncol=3)
    return fig


fig_stack = plot_drift_detection_stack(
    Path(os.environ.get("LOGDIR", "logs")) / "cldd",
    runs=[
        ("High F1", "LWF", "ADWIN", BOUNDARIES),
        ("False alarms", "LWF", "STEPD", ("gradual",)),
        ("Misses drifts", "LWF", "DDM", ("gradual",)),
    ],
    seed="005",
)
fig_stack.savefig("fig/plot_error_stream_stack.pdf")
