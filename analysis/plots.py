import os
from collections.abc import Sequence

# Run with `uv run notebook/plots.py`
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from common import (
    BOUNDARY_PALETTE,
    DETECTOR_PALETTE,
    DRIFT_WINDOW_COLOR,
    EARLY_LATE_RAW,
    FIGSIZE_SR,
    FONT_LABEL,
    FONT_SUBTITLE,
    FONT_TICK,
    FONT_TITLE,
    LABEL_ACC,
    LABEL_F1,
    LABEL_NMDT,
    MID_LABEL,
    NO_TOP_RIGHT,
    ONE_SIDED_LABELS,
    PANEL_LABEL_COLOR,
    PAPER_STYLE,
    STRATEGY_PALETTE,
    load_best_detectors,
    rename_methods,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.patheffects import withStroke
from scipy import stats

# --- Configuration & Setup ---
plt.style.use(PAPER_STYLE)
plt.rcParams.update(NO_TOP_RIGHT)

# Directory Setup
FIG_DIR = Path("fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Metrics & Constants
F1_METRIC = "dd.my_f1"
ACC_METRIC = "ocl.accuracy_seen_avg"
# The NMDT scatterplot only keeps runs with an F1 score above this
MIN_RUN_F1 = 0.5
# Alternative: only keep detectors in the top quartile of mean F1
DETECTOR_F1_QUANTILE = 0.75

STRATEGY_ORDER = ["FT", "EWC", "SI", "RWalk", "MAS", "LWF"]


# --- Helpers ---


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour with white. `amount` is the share of white (0 to 1)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def put_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    ncol: int,
    extra: Sequence[tuple[object, str]] = (),
    fontsize: float | str = "small",
    **legend_kw,
) -> None:
    """Replace the axes legend with one frameless legend below the plot.

    All figures use this so legends look the same: no title, no frame, small font,
    entries start with a capital letter. `extra` adds (handle, label) entries.
    """
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # Drop the seaborn header entries
    keep = [i for i, lbl in enumerate(labels) if lbl not in ("strategy", "boundary")]
    fig.legend(
        [handles[i] for i in keep] + [h for h, _ in extra],
        [labels[i][0].upper() + labels[i][1:] for i in keep] + [l for _, l in extra],
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        **{"columnspacing": 0.8, "handletextpad": 0.4, **legend_kw},
    )


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Helper to consistently save and close figures to manage memory."""
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def compute_nmdt(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized mean detection time (nmdt): mdt over the run's detection
    window (max_delay), so it is comparable across boundaries and datasets."""
    df = df.copy()
    df["nmdt"] = df["dd.mdt"] / df["dd.max_delay"]
    # If NAN replace with 1
    df["nmdt"] = df["nmdt"].fillna(1.0)
    return df


def prepare_best_detector_data(
    df_best: pd.DataFrame, df_ev: pd.DataFrame
) -> pd.DataFrame:
    """Keep mid, ADWIN* and the best detector of each strategy and boundary.

    :param df_best: Output of `common.load_best_detectors`. The best detector is
        picked on the tuning seeds (phase 2), not on these evaluation runs.
    """
    df_ev = df_ev.copy()

    # 1. Best detector per boundary and strategy
    best_methods_idx = pd.MultiIndex.from_frame(
        df_best[["boundary", "strategy", "best"]]
    )

    # 2. Match the best methods against df_ev using MultiIndex
    is_best_mask = df_ev.set_index(
        ["boundary", "strategy", "detector_label"]
    ).index.isin(best_methods_idx)

    # 3. Label the dynamically found best methods
    df_ev.loc[is_best_mask, "detector_label"] = "Best"

    # 4. Filter for only Best, mid, and ADWIN_JOINT
    baseline_mask = df_ev["detector_label"].isin(["oracle", "ADWIN_JOINT"])
    df_ev = df_ev[baseline_mask | is_best_mask].copy()

    # 5. Clean up naming
    df_ev["detector_label"] = df_ev["detector_label"].replace({"ADWIN_JOINT": "ADWIN*"})

    return df_ev.reset_index(drop=True)


def add_linear_fit_and_r_value(ax, df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """
    Fits a linear regression to all runs in `df`, plots the line (no confidence band) on
    the provided axes, and prints r and N on the axes. Returns the R-squared value.
    """
    data = df[[x_col, y_col]].dropna()
    x = data[x_col]
    y = data[y_col]

    # 1. Perform linear regression on all runs (no outlier filter)
    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x, y)

    # 2. Plot line across the X-range of the panel
    x_range = np.array([df[x_col].min(), df[x_col].max()])
    line = slope * x_range + intercept

    ax.plot(x_range, line, color="black", linewidth=0.8)

    # 3. Add the correlation and the number of runs to the plot
    ax.text(
        0.05,
        0.95,
        # A small table, so the = signs line up
        rf"$\begin{{array}}{{@{{}}r@{{\,}}c@{{\,}}l@{{}}}} r & = & {r_value:.2f} \\ "
        rf"N & = & {len(data)} \end{{array}}$",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=FONT_LABEL,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
    )

    return r_value**2


# --- Plotting Functions ---


def barplot_f1_detector_strategy(
    df_true: pd.DataFrame, df_sim: pd.DataFrame
) -> plt.Figure:
    """Creates a side-by-side barplot comparing true and estimated F1 scores."""
    ignore_labels = ["oracle", "ADWIN_JOINT"]
    df_true = df_true[~df_true["detector_label"].isin(ignore_labels)]
    df_sim = df_sim[~df_sim["detector_label"].isin(ignore_labels)]

    fig, (ax_true, ax_sim) = plt.subplots(
        1, 2, figsize=(FIGSIZE_LOSS[0], 1.9), sharey=True, constrained_layout=True
    )

    # Same detector order in both panels, best detector first
    order = (
        df_true.groupby("detector_label")[F1_METRIC]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # An Avg. group at the end: each strategy pooled over all detectors
    df_true = pd.concat([df_true, df_true.assign(detector_label="Avg.")])
    df_sim = pd.concat([df_sim, df_sim.assign(detector_label="Avg.")])
    order = order + ["Avg."]

    plot_kwargs = {
        "hue": "strategy",
        "y": F1_METRIC,
        "x": "detector_label",
        "order": order,
        "palette": STRATEGY_PALETTE,
        "hue_order": list(STRATEGY_PALETTE.keys()),
        "linewidth": 0,
        "errorbar": "se",
        "capsize": 0.25,
        "err_kws": {"linewidth": 0.6, "color": "black"},
    }

    sns.barplot(data=df_true, ax=ax_true, legend=False, **plot_kwargs)
    sns.barplot(data=df_sim, ax=ax_sim, legend=True, **plot_kwargs)

    # FT has no bar in (a) but keeps its slot, so move the labels to the centre of the
    # bars that are drawn: half a slot (the group is 0.8 wide) to the right
    shift = 0.8 / len(STRATEGY_PALETTE) / 2
    labels = [t.get_text() for t in ax_true.get_xticklabels()]
    ax_true.set_xticks(np.arange(len(labels)) + shift, labels)

    ax_true.set_title("(a) Active detector", fontsize=FONT_SUBTITLE)
    ax_sim.set_title("(b) Replayed error stream", fontsize=FONT_SUBTITLE)
    ax_true.set_ylabel("F1", fontsize=FONT_LABEL)
    ax_sim.set_ylabel("")
    for ax in (ax_true, ax_sim):
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="lightgrey", linewidth=0.32)
        sns.despine(ax=ax)
        ax.minorticks_off()
        ax.tick_params(axis="x", labelsize=FONT_TICK - 0.5, bottom=False, top=False)
        ax.tick_params(axis="y", labelsize=FONT_TICK, right=False)
    # (b) shares the y axis: no ticks and no edge, the grid carries the scale
    ax_sim.tick_params(axis="y", left=False)
    ax_sim.spines["left"].set_visible(False)

    put_legend(
        fig,
        ax_sim,
        ncol=6,
        fontsize=FONT_LABEL,
        handlelength=0.9,
        columnspacing=0.6,
        handletextpad=0.3,
    )

    return fig


def boxplot_detector(df: pd.DataFrame, metric: str, metric_label: str) -> plt.Figure:
    """Generates a standard boxplot for a specific metric split by boundary."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SR, constrained_layout=True)

    order = (
        df.groupby("detector_label")[metric].median().sort_values(ascending=False).index
    )

    sns.boxplot(
        data=df,
        y="detector_label",
        x=metric,
        hue="boundary",
        palette=BOUNDARY_PALETTE,
        hue_order=list(BOUNDARY_PALETTE.keys()),
        order=order,
        linewidth=0.6,
        fliersize=2.5,
        ax=ax,
    )

    ax.set(
        ylabel="", xlabel=metric_label, title=f"{metric_label} by detector and boundary"
    )
    put_legend(fig, ax, ncol=3)
    # The detector axis is categorical, so it needs no ticks
    ax.tick_params(axis="y", which="both", left=False)

    return fig


def scatter_by_boundary(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    ylim: tuple[float, float],
    share_x: bool = True,
    xlim: tuple[float, float] | None = None,
    window: dict[str, float] | None = None,
    strategy_fits: bool = False,
    references: bool = False,
    top_quantile: float | None = None,
    min_run_f1: float | None = None,
) -> tuple[plt.Figure, dict[str, float]]:
    """Scatterplot of accuracy against `x_col`, one panel per boundary.

    Colour shows the strategy. Each panel has a linear fit with its own r and N. With `strategy_fits`, each panel instead has one thin linear fit
    per strategy (no band), so a flat line shows robustness for that strategy.
    With `references`, mid detector and the FT strategy, which are left out of
    the points and fits, are drawn as horizontal lines at their mean accuracy: one
    dashed line per strategy for mid (perfect detection), and one solid grey
    line for FT (no detector).
    With `top_quantile` (for example 0.75), only detectors whose mean F1 score (over all
    strategies and boundaries) is at or above that quantile of the detector means are
    kept. The others are left out of the points and fits. The mid and FT lines stay.
    With `min_run_f1`, only runs with an F1 score above it are kept.
    `window` maps each boundary to the half-width of the acceptable window in units
    of `x_col`. It is drawn as a grey band. Points are solid dots with a small horizontal jitter so
    that stacked points stay visible (the fit uses the unjittered data).

    The y axis is zoomed to `ylim`, and the x axis to `xlim` if given. With `share_x=False` each panel zooms its own x axis
    to the 1st to 99th percentile of its data (values of 1.0, which mean
    "undefined" for NMDT, are left out of that range). Points outside are cut off but still
    count in the fit. The number of cut-off points is printed.
    """
    df = df.copy()
    df["detector_label"] = df["detector_label"].replace(
        {"ADWIN_JOINT": "ADWIN*", "oracle": MID_LABEL}
    )
    oracle_acc = (
        df[df["detector_label"] == MID_LABEL]
        .groupby(["boundary", "strategy"])[ACC_METRIC]
        .mean()
    )
    mask = (
        (df["dd.my_tp"] + df["dd.my_fp"] > 0)
        & (~df["detector_label"].isin([MID_LABEL, "ADWIN*"]))
        & (df["strategy"] != "FT")
    )
    df = df[mask]
    if top_quantile is not None:
        mean_f1 = df.groupby("detector_label")[F1_METRIC].mean()
        cutoff = mean_f1.quantile(top_quantile)
        kept = mean_f1[mean_f1 >= cutoff].index
        print(f"Mean F1 per detector: {mean_f1.round(3).to_dict()}")
        print(f"Detectors kept (mean F1 >= {cutoff:.3f}): {sorted(kept)}")
        print(f"Detectors dropped: {sorted(set(mean_f1.index) - set(kept))}")
        df = df[df["detector_label"].isin(kept)]

    if min_run_f1 is not None:
        n_before = len(df)
        df = df[df[F1_METRIC] > min_run_f1]
        print(f"Runs with F1 > {min_run_f1}: {len(df)} of {n_before}")
        print(df.groupby("detector_label").size().to_dict())

    boundaries = list(BOUNDARY_PALETTE.keys())
    fig, axes = plt.subplots(
        1,
        len(boundaries),
        figsize=(FIGSIZE_LOSS[0], 2.0),
        sharex=share_x,
        sharey=True,
        constrained_layout=True,
    )

    rng = np.random.default_rng(0)
    x_span = df[x_col].max() - df[x_col].min()
    r_squared = {}
    n_out = 0
    for idx, (ax, boundary) in enumerate(zip(axes, boundaries)):
        df_b = df[df["boundary"] == boundary].copy()
        df_b["x_jitter"] = df_b[x_col] + rng.uniform(
            -0.015 * x_span, 0.015 * x_span, len(df_b)
        )
        sns.scatterplot(
            data=df_b,
            x="x_jitter",
            y=ACC_METRIC,
            hue="strategy",
            # With strategy fit lines, tint the dots (solid, not transparent) so that
            # they sit behind the lines
            palette=(
                {k: lighten(c, 0.6) for k, c in STRATEGY_PALETTE.items()}
                if strategy_fits
                else STRATEGY_PALETTE
            ),
            hue_order=[k for k in STRATEGY_PALETTE if k != "FT"],
            marker="o",
            s=4 if strategy_fits else 6,
            linewidth=0,
            legend="auto" if idx == len(axes) - 1 else False,
            ax=ax,
        )
        if references:
            for strategy, color in STRATEGY_PALETTE.items():
                if strategy == "FT":
                    # FT has no detector, its only run is listed under mid
                    ax.axhline(
                        oracle_acc[(boundary, "FT")],
                        color=lighten(color, 0.1),
                        linewidth=0.7,
                        zorder=1,
                    )
                else:
                    # Thin, lighter reference lines stay behind the fit lines
                    ax.axhline(
                        oracle_acc[(boundary, strategy)],
                        color=lighten(color, 0.3),
                        linewidth=0.5,
                        linestyle=(0, (2, 2)),
                        zorder=1,
                    )
        if window is not None:
            ax.axvspan(
                -window[boundary],
                window[boundary],
                color=DRIFT_WINDOW_COLOR,
                alpha=0.6,
                linewidth=0,
                zorder=0,
            )
        if strategy_fits:
            valid = df_b[df_b[x_col] < 1.0]  # NMDT is 1.0 when undefined
            r_squared[boundary] = (
                stats.linregress(valid[x_col], valid[ACC_METRIC])[2] ** 2
            )
            for strategy, color in STRATEGY_PALETTE.items():
                d = valid[valid["strategy"] == strategy]
                if len(d) < 3:
                    continue
                res = stats.linregress(d[x_col], d[ACC_METRIC])
                xs = np.array([d[x_col].min(), d[x_col].max()])
                ax.plot(
                    xs,
                    res.slope * xs + res.intercept,
                    color=color,
                    linewidth=1.4,
                    zorder=3,
                    # White outline so the line stands out from the dots
                    path_effects=[withStroke(linewidth=2.6, foreground="white")],
                )
                print(
                    f"{boundary} {strategy}: slope={res.slope:.3f} p={res.pvalue:.2f}"
                )
        else:
            r_squared[boundary] = add_linear_fit_and_r_value(
                ax, df_b, x_col, ACC_METRIC
            )
        if not share_x:
            # NMDT is set to 1.0 when it is undefined, so leave those out of the range
            lo, hi = df_b.loc[df_b[x_col] < 1.0, x_col].quantile([0.01, 0.99])
            pad = 0.08 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            n_out += int(((df_b[x_col] < lo - pad) | (df_b[x_col] > hi + pad)).sum())
        ax.set_title(boundary.capitalize(), fontsize=FONT_SUBTITLE)
        ax.set_xlabel("")
        ax.set_ylabel(LABEL_ACC if idx == 0 else "", fontsize=FONT_LABEL)
        ax.minorticks_off()
        ax.tick_params(labelsize=FONT_TICK)

    axes[len(axes) // 2].set_xlabel(x_label, fontsize=FONT_LABEL)
    axes[0].set_ylim(*ylim)  # shared y axis
    out = (df[ACC_METRIC] < ylim[0]) | (df[ACC_METRIC] > ylim[1])
    print(f"{x_col}: {int(out.sum())} of {len(df)} points below or above the y range")
    if xlim is not None:
        axes[0].set_xlim(*xlim)  # shared x axis
        n_x = int(((df[x_col] < xlim[0]) | (df[x_col] > xlim[1])).sum())
        print(f"{x_col}: {n_x} of {len(df)} points outside the x range")
    if not share_x:
        print(f"{x_col}: {n_out} points outside the x range of their panel")
    extra = []
    if references:
        extra.append(
            (Line2D([], [], color=STRATEGY_PALETTE["FT"], linewidth=0.7), "FT")
        )
        extra.append(
            (
                Line2D([], [], color="dimgrey", linewidth=0.5, linestyle=(0, (2, 2))),
                MID_LABEL,
            )
        )
    if window is not None:
        extra.append(
            (Patch(facecolor=DRIFT_WINDOW_COLOR, alpha=0.6), "Acceptable window")
        )
    put_legend(
        fig,
        axes[-1],
        ncol=4 if extra else 5,
        extra=extra,
        fontsize=FONT_LABEL,
        handlelength=0.9,
        columnspacing=0.6,
        handletextpad=0.3,
    )
    return fig, r_squared


def scatterplot_acc_f1_strategy_boundary(
    df_true: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, float]]:
    """Creates a scatterplot comparing Accuracy vs. F1 score."""
    return scatter_by_boundary(df_true, F1_METRIC, "F1", ylim=(0.45, 0.76))


def scatterplot_acc_nmdt_strategy_boundary(
    df_true: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, float]]:
    """Creates a scatterplot comparing Accuracy vs. normalized mean detection time."""
    return scatter_by_boundary(
        compute_nmdt(df_true),
        "nmdt",
        LABEL_NMDT,
        ylim=(0.45, 0.76),
        xlim=(-1, 1),
        strategy_fits=True,
        min_run_f1=MIN_RUN_F1,
    )


def lowess(x: np.ndarray, y: np.ndarray, grid: np.ndarray, frac: float) -> np.ndarray:
    """Local linear fit with tricube weights over the `frac` nearest points."""
    k = max(int(np.ceil(frac * len(x))), 3)
    out = np.full(len(grid), np.nan)
    for i, x0 in enumerate(grid):
        d = np.abs(x - x0)
        h = max(np.sort(d)[k - 1], 1e-9)
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        sw = np.sqrt(w)
        design = np.column_stack([np.ones_like(x), x - x0]) * sw[:, None]
        if (w > 0).sum() >= 3:
            out[i] = np.linalg.lstsq(design, y * sw, rcond=None)[0][0]
    return out


def add_acc_loss(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Add NMDT and `loss`: accuracy minus mid run with the same strategy,
    boundary and seed."""
    df = compute_nmdt(df_eval)
    df["detector_label"] = df["detector_label"].replace({"ADWIN_JOINT": "ADWIN*"})
    oracle = df[df["detector_label"] == "oracle"].set_index(
        ["boundary", "strategy", "seed"]
    )[ACC_METRIC]
    assert oracle.index.is_unique
    df["loss"] = df[ACC_METRIC] - [
        oracle[key] for key in zip(df["boundary"], df["strategy"], df["seed"])
    ]
    return df


# Narrower than a column: the saved PDF adds padding and must stay within 252 pt
FIGSIZE_LOSS = (248 / 72, 2.5)


def lineplot_acc_loss_strategy_boundary(
    df_eval: pd.DataFrame,
    x_col: str,
    x_label: str,
    edges: Sequence[float],
    n_min: int = 3,
    x_at_bin_mean: bool = False,
    skip_boundaries: Sequence[str] = (),
    title: str = "",
    smooth: str = "bins",
    max_fp: int | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Accuracy minus control against `x_col`, one panel per boundary.

    Y is the accuracy of a run minus the accuracy of the control (mid) run with the
    same strategy, boundary and seed, so 0 means "as good as perfect detection". `x_col` is
    split into the bins given by `edges`, and a bin is drawn at its centre, or at the
    mean of its runs with `x_at_bin_mean`. Each strategy shows the mean per bin. All detectors are used, but runs with an undefined
    NMDT are left out (for "nmdt"). Bins with fewer than `n_min` runs are not drawn.
    With `smooth="lowess"`, each strategy instead shows a lowess curve over the single
    runs (no bins).

    With `max_fp`, only runs with at most that many false alarms are used.

    Boundaries in `skip_boundaries` get no panel.

    Returns the figure and the table of binned means.
    """
    df = add_acc_loss(df_eval)
    edges = np.asarray(edges, dtype=float)

    # Real detectors on defined NMDT, as in the other NMDT figure
    df = df[
        (~df["detector_label"].isin(["oracle", "ADWIN*"]))
        & (df["strategy"] != "FT")
        & (df["dd.mdt"].notna() | (x_col != "nmdt"))
        & (df["dd.my_tp"] + df["dd.my_fp"] > 0)
        & ((df["dd.my_fp"] <= max_fp) if max_fp is not None else True)
    ].copy()

    centres = (edges[:-1] + edges[1:]) / 2
    bin_idx = pd.cut(df[x_col], edges, labels=False, include_lowest=True)
    if x_at_bin_mean:
        df["x"] = df.groupby(bin_idx)[x_col].transform("mean")
    else:
        df["x"] = bin_idx.map(lambda i: centres[int(i)] if pd.notna(i) else np.nan)
    x_lo, x_hi = edges[0], edges[-1]
    pad = 0.02 * (x_hi - x_lo)

    boundaries = [b for b in BOUNDARY_PALETTE if b not in skip_boundaries]
    ylim = (-0.2, 0.1)
    fig, axes = plt.subplots(
        1,
        len(boundaries),
        figsize=FIGSIZE_LOSS,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    rng = np.random.default_rng(0)
    tables = []
    n_out = 0
    for idx, (ax, boundary) in enumerate(zip(axes, boundaries)):
        df_b = df[df["boundary"] == boundary]
        ax.axhline(0, color="black", linewidth=0.5, zorder=1)
        # Light single runs behind the lines
        pts = df_b[df_b[x_col].between(x_lo, x_hi)]
        jitter = 0.01 * (x_hi - x_lo)
        ax.scatter(
            pts[x_col] + rng.uniform(-jitter, jitter, len(pts)),
            pts["loss"],
            s=3,
            linewidth=0,
            zorder=2,
            color=[lighten(STRATEGY_PALETTE[st], 0.6) for st in pts["strategy"]],
        )
        for strategy, color in STRATEGY_PALETTE.items():
            if strategy == "FT":
                continue
            if smooth == "lowess":
                g = df_b[df_b["strategy"] == strategy].dropna(subset=[x_col])
                gx, gy = g[x_col].to_numpy(float), g["loss"].to_numpy(float)
                if len(g) < 3 * n_min:
                    continue
                grid = np.linspace(*np.percentile(gx, [5, 95]), 25)
                mean = lowess(gx, gy, grid, frac=0.9)
                tables.append(
                    pd.DataFrame(
                        {
                            "boundary": boundary,
                            "strategy": strategy,
                            "x": grid,
                            "mean": mean,
                        }
                    )
                )
                x_line, y_line = grid, mean
                marker = None
            else:
                g = df_b[df_b["strategy"] == strategy].dropna(subset=["x"])
                agg = (
                    g.groupby("x", observed=True)["loss"]
                    .agg(["mean", "sem", "count"])
                    .reset_index()
                )
                agg = agg[agg["count"] >= n_min]
                # 95% confidence interval of the mean (t distribution)
                agg["ci95"] = agg["sem"] * stats.t.ppf(0.975, agg["count"] - 1)
                agg.insert(0, "strategy", strategy)
                agg.insert(0, "boundary", boundary)
                tables.append(agg)
                x_line, y_line = agg["x"], agg["mean"]
                marker = "o"
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=0.9,
                marker=marker,
                markersize=3,
                solid_capstyle="round",
                zorder=3,
                label=strategy,
                path_effects=[
                    withStroke(linewidth=1.9, foreground="white", capstyle="round")
                ],
            )
        n_out += int((df_b["loss"] < ylim[0]).sum() + (df_b["loss"] > ylim[1]).sum())
        ax.set_title(boundary.capitalize(), fontsize=FONT_SUBTITLE)
        ax.set_xlabel("")
        ax.set_ylabel("Accuracy $-$ control" if idx == 0 else "", fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK)
    axes[0].set_xlim(x_lo - pad, x_hi + pad)
    axes[0].set_ylim(*ylim)
    # Name the sides of the zero line at the top and bottom of the first panel
    for text, y, va, dy in (
        ("$>$ control", 1, "top", -3),
        ("$<$ control", 0, "bottom", 3),
    ):
        axes[0].annotate(
            text,
            xy=(0.04, y),
            xycoords="axes fraction",
            xytext=(2, dy),
            textcoords="offset points",
            ha="left",
            va=va,
            fontsize=FONT_SUBTITLE,
            color="dimgrey",
        )
    axes[len(axes) // 2].set_xlabel(x_label, fontsize=FONT_LABEL)
    axes[0].set_xticks([x_lo, (x_lo + x_hi) / 2, x_hi])
    axes[0].set_yticks([-0.2, -0.1, 0, 0.1])
    print(f"loss plot {x_col}: {n_out} of {len(df)} runs outside the y range")
    put_legend(
        fig,
        axes[-1],
        ncol=len(STRATEGY_PALETTE) - 1,
        fontsize=FONT_LABEL,
        handlelength=0.9,
        columnspacing=0.3,
        handletextpad=0.2,
    )
    if title:
        fig.suptitle(rf"\textbf{{{title}}}", x=0, ha="left", fontsize=FONT_TITLE)
    return fig, pd.concat(tables, ignore_index=True)


# Letters that the caption refers to: (letter, NMDT, F1) in each panel
CONTOUR_LETTERS = {
    "abrupt": ("(a)", -0.55, 0.5),
    "gradual": ("(b)", 0.3, 0.62),
    "slow": ("(c)", -0.55, 0.9),
}


def contour_acc_nmdt_f1_boundary(
    df_eval: pd.DataFrame,
    bandwidth: tuple[float, float] = (0.15, 0.12),
    min_weight: float = 5.0,
) -> plt.Figure:
    """Accuracy over NMDT (x) and detector F1 (y), square panels at single-column width.

    Colour is the Gaussian-weighted mean accuracy of all strategies at each point, with the
    `bandwidth` (NMDT, F1). Points with less than `min_weight` runs nearby are left
    blank. Black dots are the single runs.
    """
    df = add_acc_loss(df_eval)
    df = df[
        (~df["detector_label"].isin(["oracle", "ADWIN*"]))
        & (df["strategy"] != "FT")
        & df["dd.mdt"].notna()
        & (df["dd.my_tp"] + df["dd.my_fp"] > 0)
    ]
    gx, gy = np.meshgrid(np.linspace(-1, 1, 300), np.linspace(0, 1, 300))
    fig, axes = plt.subplots(
        1,
        len(BOUNDARY_PALETTE),
        figsize=(240 / 72, 1.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    runs, means = {}, {}
    for boundary in BOUNDARY_PALETTE:
        g = df[df["boundary"] == boundary]
        x, y, z = (
            g["nmdt"].to_numpy(),
            g[F1_METRIC].to_numpy(),
            g[ACC_METRIC].to_numpy(),
        )
        w = np.exp(
            -0.5
            * (
                ((gx[..., None] - x) / bandwidth[0]) ** 2
                + ((gy[..., None] - y) / bandwidth[1]) ** 2
            )
        )
        weight = w.sum(-1)
        runs[boundary] = (x, y)
        means[boundary] = np.ma.masked_where(
            weight < min_weight, (w * z).sum(-1) / weight
        )
    # One scale for all panels, from the lowest to the highest value drawn, in steps of 0.02
    lo = np.floor(min(m.min() for m in means.values()) * 50) / 50
    hi = np.ceil(max(m.max() for m in means.values()) * 50) / 50
    levels = np.linspace(lo, hi, round((hi - lo) * 50) + 1)
    for letter, ax, boundary in zip("abc", axes, BOUNDARY_PALETTE):
        cs = ax.contourf(gx, gy, means[boundary], levels=levels, cmap=cm.batlow)
        # Draw each band edge in its own colour so no white seams show between bands
        cs.set_edgecolor("face")
        cs.set_linewidth(0.3)
        ax.scatter(*runs[boundary], s=2.5, color="black", linewidth=0, zorder=3)
        letter, lx, ly = CONTOUR_LETTERS[boundary]
        ax.text(
            lx,
            ly,
            rf"\textbf{{{letter}}}",
            color=PANEL_LABEL_COLOR,
            fontsize=FONT_LABEL,
            ha="center",
            va="center",
            zorder=4,
            path_effects=[withStroke(linewidth=1.5, foreground="white")],
        )
        ax.set_title(boundary.capitalize(), fontsize=FONT_SUBTITLE)
        ax.minorticks_off()
        ax.tick_params(labelsize=FONT_TICK, top=True, right=True)
        ax.set_xticks([-1, 0, 1])
        ax.set_box_aspect(1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    axes[0].set_ylabel("F1", fontsize=FONT_LABEL)
    axes[len(axes) // 2].set_xlabel(f"{LABEL_NMDT} (delay)", fontsize=FONT_LABEL)
    cbar = fig.colorbar(
        cs,
        cax=axes[-1].inset_axes([1.12, 0, 0.08, 1]),
        orientation="vertical",
        ticks=np.round(np.arange(np.ceil(lo * 10) / 10, hi, 0.1), 1),
    )
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=FONT_TICK)
    # Vertical label beside the bar, between the 0.5 and 0.6 ticks
    cbar.ax.annotate(
        "Acc.",
        xy=(1, 0.55),
        xycoords=cbar.ax.get_yaxis_transform(),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        rotation=90,
        fontsize=FONT_LABEL,
    )
    return fig


DATASET_LABELS = {"rfmnist": "RFMNIST", "covertype": "ECovtype"}


HEATMAP_BASELINES = ["early", MID_LABEL, "late"]


def heatmap_strategy_detector_boundary(
    df_evals: dict[str, pd.DataFrame],
    metric: str,
    title: str,
    reference: bool,
) -> plt.Figure:
    """Mean `metric` (shown x100) for each strategy (rows) and detector (columns).

    With `reference` the early, mid and late baselines are the first columns and FT the
    last row. Without it (drift detection metrics, which they do not have) they are left
    out.

    One row of panels per dataset and one column per boundary. FT has no detector, so its
    accuracy spans the baseline and detector columns in the last row (it runs once per
    boundary), stopping before the Avg. column. The FT row is not part of any average.
    Every cell shows its value, so there is no colour bar. The colour scale is
    shared by the panels of a dataset. Within a dataset, rows and columns are sorted from
    best to worst by their mean over all panels, with the baselines first. The last column and
    row are the mean of the cells in their row or column. The corner is the mean over all
    strategies and detectors.
    """
    strategies = [st for st in STRATEGY_PALETTE if st != "FT"]
    raw: dict[str, tuple[dict[str, pd.DataFrame], pd.Series]] = {}
    for name, df_eval in df_evals.items():
        df = df_eval.replace(rename_methods)
        ft = df[df["strategy"] == "FT"].groupby("boundary")[metric].mean()
        if not reference:
            ft = pd.Series(dtype=float)
        raw[name] = (
            {
                b: df[(df["boundary"] == b) & (df["strategy"] != "FT")]
                .pipe(lambda d: d[d["detector_label"] != "BEST"])
                .groupby(["strategy", "detector_label"])[metric]
                .mean()
                .unstack()
                for b in BOUNDARY_PALETTE
            },
            ft,
        )
    # Columns are shared by all panels: the union of the detectors, best first by the mean
    # over all panels of all datasets. The baselines stay first.
    others = sorted(
        {d for tables, _ in raw.values() for t in tables.values() for d in t.columns}
        - set(HEATMAP_BASELINES)
    )
    overalls = [
        (sum(t.reindex(index=strategies, columns=others) for t in tables.values()))
        / len(tables)
        for tables, _ in raw.values()
    ]
    detectors = (HEATMAP_BASELINES if reference else []) + (
        pd.concat([o.mean() for o in overalls], axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )
    all_tables: dict[str, dict[str, pd.DataFrame]] = {}
    norms = {}
    for name, (tables, ft) in raw.items():
        tables = {
            b: t.reindex(index=strategies, columns=detectors) for b, t in tables.items()
        }
        lo = min([t.min().min() for t in tables.values()] + list(ft))
        hi = max([t.max().max() for t in tables.values()] + list(ft))
        lo, hi = np.floor(lo * 50) / 50, np.ceil(hi * 50) / 50
        # Same row order in every panel of a dataset: best first, by the mean over all panels
        overall = sum(tables.values()) / len(tables)
        order = overall.mean(axis=1).sort_values(ascending=False).index.tolist()
        tables = {b: t.reindex(index=order) for b, t in tables.items()}
        for t in tables.values():
            t["Avg."] = t.mean(axis=1)
            t.loc["Avg."] = t.mean(axis=0)
        if reference:
            for b, t in tables.items():
                t.loc["FT"] = np.nan
                t.loc["FT", MID_LABEL] = ft[b]
        all_tables[name] = tables
        norms[name] = mcolors.Normalize(lo, hi)
    gap = 0.35  # Extra space before the Avg. row and column, in cells
    # Size the figure so the equal-aspect panels fill their grid cells with no slack
    n_rows, n_cols = next(iter(next(iter(all_tables.values())).values())).shape
    left, right, wspace = 0.1, 0.99, 0.12
    width = FIGSIZE_LOSS[0]
    panel_w = width * (right - left) / (len(BOUNDARY_PALETTE) * (1 + wspace) - wspace)
    n_gaps = 2 if reference else 1  # before the Avg. row, and before the FT row
    panel_h = panel_w * (n_rows + n_gaps * gap) / (n_cols + gap)
    head, foot, between = 0.45, 0.5, 0.12  # inches
    n_data = len(df_evals)
    height = n_data * panel_h + (n_data - 1) * between + head + foot
    fig, axes = plt.subplots(
        n_data,
        len(BOUNDARY_PALETTE),
        figsize=(width, height),
        sharey="row",
        squeeze=False,
    )
    fig.subplots_adjust(
        left=left,
        right=right,
        top=1 - head / height,
        bottom=foot / height,
        wspace=wspace,
        hspace=between / panel_h,
    )
    panels = [
        (axes[r, c], name, boundary, table)
        for r, (name, tables) in enumerate(all_tables.items())
        for c, (boundary, table) in enumerate(tables.items())
    ]
    for ax, name, boundary, table in panels:
        norm = norms[name]
        n_rows, n_cols = table.shape
        xs = np.arange(n_cols) + gap * (np.arange(n_cols) == n_cols - 1)
        # Gap before the Avg. row, and another before the FT row
        ys = np.arange(n_rows) + gap * (np.arange(n_rows) >= n_rows - n_gaps)
        ys = ys + gap * (reference and np.arange(n_rows) == n_rows - 1)
        for i, y in enumerate(ys):
            if reference and table.index[i] == "FT":
                v = table.loc["FT", MID_LABEL]
                colour = cm.batlow(norm(v))
                ax.add_patch(
                    Rectangle(
                        (-0.5, y - 0.5),
                        xs[-2] + 1,
                        1,
                        facecolor=colour,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                text = f"{100 * v:.0f}"
                ax.annotate(
                    text,
                    xy=(-0.4, y),
                    xytext=(0, -0.36 * FONT_TICK),
                    textcoords="offset points",
                    ha="left",
                    va="baseline",
                    fontsize=FONT_TICK,
                    color="white" if np.mean(colour[:3]) < 0.5 else "black",
                )
                continue
            for j, x in enumerate(xs):
                v = table.iat[i, j]
                if np.isnan(v):
                    continue
                colour = cm.batlow(norm(v))
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor=colour,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                )
                # Digits have no descender, so centre them by half their height (0.72 em)
                text = f"{100 * v:.0f}"
                # Three digits (100) are shrunk to fit the cell
                size = FONT_TICK * min(1.0, 8 / n_cols)
                if len(text) == 3:
                    size *= 0.75
                ax.annotate(
                    text,
                    xy=(x, y),
                    xytext=(0, -0.36 * size),
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                    fontsize=size,
                    color="white" if np.mean(colour[:3]) < 0.5 else "black",
                )
        ax.set_xlim(-0.5, xs[-1] + 0.5)
        ax.set_ylim(ys[-1] + 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(
            xs,
            [
                f"Oracle ({label})" if label in HEATMAP_BASELINES else label
                for label in table.columns
            ]
            if ax.get_subplotspec().is_last_row()
            else [],
            rotation=90,
            fontsize=FONT_TICK,
        )
        ax.set_yticks(ys, table.index, fontsize=FONT_TICK)
        ax.tick_params(which="both", length=0)
        if ax.get_subplotspec().is_first_row():
            ax.set_title(boundary.capitalize(), fontsize=FONT_SUBTITLE)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(DATASET_LABELS.get(name, name), fontsize=FONT_LABEL)
        for spine in ax.spines.values():
            spine.set_visible(False)
    # One label for all cells, centred over the middle column
    fig.canvas.draw()
    middle = axes[0, len(BOUNDARY_PALETTE) // 2].get_position()
    fig.suptitle(
        title,
        x=(middle.x0 + middle.x1) / 2,
        y=1.0,
        ha="center",
        fontsize=FONT_SUBTITLE,
    )
    return fig


def heatmap_acc_strategy_detector_boundary(
    df_evals: dict[str, pd.DataFrame],
) -> plt.Figure:
    return heatmap_strategy_detector_boundary(
        df_evals, ACC_METRIC, "Accuracy (\\%)", reference=True
    )


def heatmap_f1_strategy_detector_boundary(
    df_evals: dict[str, pd.DataFrame],
) -> plt.Figure:
    return heatmap_strategy_detector_boundary(
        df_evals, F1_METRIC, "F1 (\\%)", reference=False
    )


def lineplot_acc_loss_nmdt_strategy_boundary(df_eval: pd.DataFrame):
    """Accuracy relative to mid against NMDT, one panel per drift speed."""
    return lineplot_acc_loss_strategy_boundary(
        df_eval,
        "nmdt",
        LABEL_NMDT,
        np.linspace(-1, 1, 5),
        n_min=5,
        smooth="lowess",
    )


def lineplot_acc_loss_f1_strategy_boundary(df_eval: pd.DataFrame):
    """Accuracy relative to mid against F1 score."""
    return lineplot_acc_loss_strategy_boundary(
        df_eval, F1_METRIC, LABEL_F1, np.linspace(0, 1, 5), x_at_bin_mean=True
    )


def plot_faceted_strategy_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Generates a grid of boxplots for accuracy across strategies and boundary types."""
    boundaries = sorted(df["boundary"].dropna().unique())
    n_bounds = len(boundaries)

    fig, axes = plt.subplots(
        1, n_bounds, figsize=FIGSIZE_SR, sharey=True, constrained_layout=True
    )

    if n_bounds == 1:
        axes = [axes]

    for idx, (ax, boundary) in enumerate(zip(axes, boundaries)):
        is_last = idx == (n_bounds - 1)
        mask = df["boundary"] == boundary

        sns.boxplot(
            data=df[mask],
            x="strategy",
            y=ACC_METRIC,
            hue="detector_label",
            ax=ax,
            palette=DETECTOR_PALETTE,
            order=STRATEGY_ORDER,
            legend=is_last,
            linewidth=0.6,
            fliersize=2.5,
        )

        ax.set(
            title=f"{boundary.capitalize()} boundary",
            xlabel="",
            ylabel=LABEL_ACC if idx == 0 else "",
        )
        ax.tick_params(axis="x", which="both", rotation=45, bottom=False)
        ax.xaxis.set_minor_locator(plt.NullLocator())

    put_legend(fig, axes[-1], ncol=3)

    return fig


if __name__ == "__main__":
    # 1. Load data
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd" / "rfmnist"
    df_eval_raw = pd.read_csv(logs_path / "evaluate/data_frame.csv")
    df_dd_raw = pd.read_csv(logs_path / "dd_run/data_frame.csv")
    for _df in (df_eval_raw, df_dd_raw):
        _df["detector_label"] = _df["detector_label"].replace(ONE_SIDED_LABELS)
    # Early and late are baselines without drift detection scores. Only the accuracy
    # heatmap shows them.
    df_eval_all = df_eval_raw
    df_eval_raw = df_eval_all[~df_eval_all["detector_label"].isin(EARLY_LATE_RAW)]

    # 2. Generate standard multi-variable plots
    fig_bar = barplot_f1_detector_strategy(df_eval_raw, df_dd_raw)
    save_fig(fig_bar, "barplot_f1_detector_strategy.pdf")

    fig_scatter, r2_f1 = scatterplot_acc_f1_strategy_boundary(df_eval_raw)
    save_fig(fig_scatter, "scatterplot_acc_f1_strategy_boundary.pdf")
    print(
        f"scatterplot_acc_f1_strategy_boundary: R^2={ {k: round(float(v), 3) for k, v in r2_f1.items()} }"
    )

    fig_scatter_nmdt, r2_nmdt = scatterplot_acc_nmdt_strategy_boundary(df_eval_raw)
    save_fig(fig_scatter_nmdt, "scatterplot_acc_nmdt_strategy_boundary.pdf")
    print(
        f"scatterplot_acc_nmdt_strategy_boundary: R^2={ {k: round(float(v), 3) for k, v in r2_nmdt.items()} }"
    )

    df_eval_covertype = pd.read_csv(
        logs_path.parent / "covertype" / "evaluate/data_frame.csv"
    )
    df_eval_covertype["detector_label"] = df_eval_covertype["detector_label"].replace(
        ONE_SIDED_LABELS
    )
    heat_data = {"rfmnist": df_eval_all, "covertype": df_eval_covertype}
    fig_heat = heatmap_acc_strategy_detector_boundary(heat_data)
    save_fig(fig_heat, "hm_acc.pdf")
    fig_heat_f1 = heatmap_f1_strategy_detector_boundary(heat_data)
    save_fig(fig_heat_f1, "hm_f1.pdf")

    fig_loss, loss_table = lineplot_acc_loss_nmdt_strategy_boundary(df_eval_raw)
    save_fig(fig_loss, "lineplot_accloss_nmdt_strategy_boundary.pdf")
    print(loss_table.round(3).to_string())

    fig_contour = contour_acc_nmdt_f1_boundary(df_eval_raw)
    save_fig(fig_contour, "contour_acc.pdf")

    fig_loss_f1, loss_f1_table = lineplot_acc_loss_f1_strategy_boundary(df_eval_raw)
    save_fig(fig_loss_f1, "lineplot_accloss_f1_strategy_boundary.pdf")
    print(loss_f1_table.round(3).to_string())

    # 3. Generate individual detector boxplots
    df_box = df_eval_raw.replace(rename_methods).pipe(compute_nmdt)
    mask_not_oracle = df_box["detector"] != MID_LABEL

    fig_acc = boxplot_detector(df_box, ACC_METRIC, LABEL_ACC)
    save_fig(fig_acc, "boxplot_detector_acc.pdf")

    fig_f1 = boxplot_detector(df_box[mask_not_oracle], F1_METRIC, LABEL_F1)
    save_fig(fig_f1, "boxplot_detector_f1.pdf")

    fig_nmdt = boxplot_detector(df_box[mask_not_oracle], "nmdt", LABEL_NMDT)
    save_fig(fig_nmdt, "boxplot_detector_nmdt.pdf")

    # 4. Generate faceted strategy performance boxplots
    df_faceted = prepare_best_detector_data(load_best_detectors(logs_path), df_eval_raw)
    fig_faceted = plot_faceted_strategy_boxplots(df_faceted)
    save_fig(fig_faceted, "boxplot_accuracy_strategy_detector.pdf")
