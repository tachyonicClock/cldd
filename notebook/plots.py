# Run with `uv run notebook/plots.py`
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots as _  # noqa: F401

from common import FIGSIZE_SR, atab10, rename_methods

# --- Configuration & Setup ---
plt.style.use(["science", "nature"])

# Directory Setup
FIG_DIR = Path("fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Metrics & Constants
F1_METRIC = "dd.my_f1"
ACC_METRIC = "ocl.accuracy_seen_avg"
BOUNDARY_DIVISORS = {"abrupt": 640, "gradual": 12_000, "slow": 24_000}

STRATEGY_ORDER = ["FT", "EWC", "SI", "RWalk", "MAS", "LWF"]
DETECTOR_PALETTE = {
    "oracle": atab10(5),
    "ADWIN*": atab10(7),
    "Best": atab10(9),
}


# --- Helpers ---


def save_fig(fig: plt.Figure, filename: str) -> None:
    """Helper to consistently save and close figures to manage memory."""
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def compute_nmdt(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized mean detection time (nmdt) based on boundary type."""
    df = df.copy()
    df["nmdt"] = df["dd.mdt"] / df["boundary"].map(BOUNDARY_DIVISORS)
    return df


def prepare_best_detector_data(
    df_dd: pd.DataFrame, df_ev: pd.DataFrame
) -> pd.DataFrame:
    """Filters data and identifies the best detectors across strategies for faceted boxplots."""
    df_ev = df_ev.copy()

    # 1. Identify best methods dynamically
    best_methods_idx = (
        df_dd.groupby(["boundary", "strategy", "detector"])[F1_METRIC]
        .mean()
        .groupby(["boundary", "strategy"])
        .idxmax()
    )

    # 2. Match the best methods against df_ev using MultiIndex
    is_best_mask = df_ev.set_index(
        ["boundary", "strategy", "detector_label"]
    ).index.isin(best_methods_idx)

    # 3. Label the dynamically found best methods
    df_ev.loc[is_best_mask, "detector_label"] = "Best"

    # 4. Filter for only Best, Oracle, and ADWIN_JOINT
    baseline_mask = df_ev["detector_label"].isin(["oracle", "ADWIN_JOINT"])
    df_ev = df_ev[baseline_mask | is_best_mask].copy()

    # 5. Clean up naming
    df_ev["detector_label"] = df_ev["detector_label"].replace({"ADWIN_JOINT": "ADWIN*"})

    return df_ev.reset_index(drop=True)


# --- Plotting Functions ---


def barplot_f1_detector_strategy(
    df_true: pd.DataFrame, df_sim: pd.DataFrame
) -> plt.Figure:
    """Creates a side-by-side barplot comparing true and estimated F1 scores."""
    ignore_labels = ["oracle", "ADWIN_JOINT"]
    df_true = df_true[~df_true["detector_label"].isin(ignore_labels)]
    df_sim = df_sim[~df_sim["detector_label"].isin(ignore_labels)]

    fig, (ax_true, ax_sim) = plt.subplots(
        1, 2, figsize=FIGSIZE_SR, sharey=True, constrained_layout=True
    )

    order = df_true.groupby("detector_label")[F1_METRIC].median().sort_values().index

    plot_kwargs = {
        "hue": "strategy",
        "y": F1_METRIC,
        "x": "detector_label",
        "palette": atab10.colors,
        "order": order,
        "edgecolor": "dimgrey",
    }

    sns.barplot(data=df_true, ax=ax_true, legend=False, **plot_kwargs)
    sns.barplot(data=df_sim, ax=ax_sim, legend=True, **plot_kwargs)

    ax_true.set(title="True Performance", xlabel="", ylabel="F1 score")
    ax_sim.set(title="Estimated Performance", xlabel="", ylabel="F1 score")

    for ax in (ax_true, ax_sim):
        ax.tick_params(axis="x", rotation=45)

    sns.move_legend(ax_sim, "upper left", title="", frameon=True, framealpha=1.0)

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
        palette=atab10.colors[1::3],
        order=order,
        ax=ax,
    )

    ax.set(
        ylabel="", xlabel=metric_label, title=f"{metric_label} by Detector and Boundary"
    )
    sns.move_legend(ax, "best", title="", frameon=True)

    return fig


def scatterplot_acc_f1_strategy_boundary(df_true: pd.DataFrame) -> plt.Figure:
    """Creates a scatterplot comparing Accuracy vs. F1 score."""
    df_filtered = df_true.copy()
    df_filtered["detector_label"] = df_filtered["detector_label"].replace(
        {"ADWIN_JOINT": "ADWIN*", "oracle": "Oracle"}
    )

    mask = (
        (df_filtered["dd.my_tp"] + df_filtered["dd.my_fp"] > 0)
        & (~df_filtered["detector_label"].isin(["Oracle", "ADWIN*"]))
        & (df_filtered["strategy"] != "FT")
    )
    df_filtered = df_filtered[mask]

    fig, ax = plt.subplots(figsize=FIGSIZE_SR, constrained_layout=True)

    sns.scatterplot(
        data=df_filtered,
        hue="strategy",
        x=F1_METRIC,
        y=ACC_METRIC,
        style="boundary",
        palette=atab10.colors,
        markers=["o", "s", "X"],
        s=15,
        ax=ax,
    )

    ax.set(
        xlabel="F1",
        ylabel="Acc.",
        title="Acc. vs F1 by Strategy and Boundary Type",
    )

    handles, labels = ax.get_legend_handles_labels()
    label_map = {"strategy": "Strategy", "boundary": "Boundary"}
    labels = [label_map.get(lbl, lbl) for lbl in labels]

    ax.legend(
        handles=handles,
        labels=labels,
        title="",
        fontsize="small",
        loc="lower right",
        ncol=2,
    )

    return fig


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
        )

        ax.set(
            title=f"{boundary.capitalize()} Boundary",
            xlabel="",
            ylabel=r"Accuracy Seen Avg. $\uparrow$" if idx == 0 else "",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_minor_locator(plt.NullLocator())

    if axes[-1].legend_ is not None:
        axes[-1].legend_.set_title("Detector")

    return fig


if __name__ == "__main__":
    # 1. Load data
    logs_path = Path("logs")
    df_eval_raw = pd.read_csv(logs_path / "evaluate/data_frame.csv")
    df_dd_raw = pd.read_csv(logs_path / "dd_run/data_frame.csv")

    # 2. Generate standard multi-variable plots
    fig_bar = barplot_f1_detector_strategy(df_eval_raw, df_dd_raw)
    save_fig(fig_bar, "barplot_f1_detector_strategy.pdf")

    fig_scatter = scatterplot_acc_f1_strategy_boundary(df_eval_raw)
    save_fig(fig_scatter, "scatterplot_acc_f1_strategy_boundary.pdf")

    # 3. Generate individual detector boxplots
    df_box = df_eval_raw.replace(rename_methods).pipe(compute_nmdt)
    mask_not_oracle = df_box["detector"] != "Oracle"

    fig_acc = boxplot_detector(df_box, ACC_METRIC, "Acc.")
    save_fig(fig_acc, "boxplot_detector_acc.pdf")

    fig_f1 = boxplot_detector(df_box[mask_not_oracle], F1_METRIC, "F1 Score")
    save_fig(fig_f1, "boxplot_detector_f1.pdf")

    fig_nmdt = boxplot_detector(df_box[mask_not_oracle], "nmdt", "Normalized MDT")
    save_fig(fig_nmdt, "boxplot_detector_nmdt.pdf")

    # 4. Generate faceted strategy performance boxplots
    df_faceted = prepare_best_detector_data(df_dd_raw, df_eval_raw)
    fig_faceted = plot_faceted_strategy_boxplots(df_faceted)
    save_fig(fig_faceted, "boxplot_accuracy_strategy_detector.pdf")
