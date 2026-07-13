# Run with `uv run notebook/plot_error_stream.py`
import pickle
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from common import FIGSIZE_43
import scienceplots as _  # noqa: F401
plt.style.use(["science", "nature"])

# %%
METRIC_FONTSIZE = 8

def plot_error_stream(metrics, ax: plt.Axes, accuracy_seen_avg: float | None = None):

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

    ax.plot(xs, ys, color="black")

    # Vertical lines at preds
    ax.vlines(trues, ymin=0, ymax=1, color="darkgreen", label="True Drift")
    ax.vlines(
        preds, ymin=0, ymax=1, color="darkred", linestyles="--", label="Predicted Drift"
    )
    # Fill between the trues - max_early and trues + max_delay
    for true in trues:
        ax.fill_betweenx(
            [0, 1], true - max_early, true + max_delay, color="lightgrey", alpha=1
        )

    # Add Labels for the metrics
    ax.text(
        0.83,
        0.95,
        f"Acc: {accuracy_seen_avg*100:.1f}%\nF1: {f1*100:.1f}%\nTP/FP/FN: {tp}/{fp}/{fn}",
        transform=ax.transAxes,
        fontsize=METRIC_FONTSIZE,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="dimgrey", boxstyle="round,pad=0.5,rounding_size=0.1"),
    )

def load_and_plot_error_stream(metrics_path: Path | str, ax: plt.Axes):
    metrics_path = Path(metrics_path)
    dd_metrics = pickle.load(open(metrics_path / "dd_metrics.pkl", "rb"))
    ocl_metrics = pickle.load(open(metrics_path / "ocl_metrics.pkl", "rb"))
    accuracy_seen_avg = ocl_metrics.accuracy_seen_avg 
    plot_error_stream(dd_metrics, ax, accuracy_seen_avg=accuracy_seen_avg)



def plot_drift_detection(root: Path, strategy: str = "EWC", detector: str = "ADWIN", seed: str = "000") -> plt.Figure:
    fig, axs = plt.subplots(figsize=FIGSIZE_43, constrained_layout=True, nrows=3, sharex=True, sharey=True)
    load_and_plot_error_stream(root / f"evaluate/abrupt/{strategy}/{detector}/{seed}/", axs[0])
    load_and_plot_error_stream(root / f"evaluate/gradual/{strategy}/{detector}/{seed}/", axs[1])
    load_and_plot_error_stream(root / f"evaluate/slow/{strategy}/{detector}/{seed}/", axs[2])

    fig.suptitle(f"Drift Detection with {detector} for {strategy}")

    axs[0].set_title("abrupt", fontsize=METRIC_FONTSIZE)
    axs[1].set_title("gradual", fontsize=METRIC_FONTSIZE)
    axs[2].set_title("slow", fontsize=METRIC_FONTSIZE)

    axs[1].set_ylabel("Error Rate")
    axs[2].set_xlabel("Stream Index")
    return fig

good_fig = plot_drift_detection(Path("logs"), strategy="LWF", detector="ADWIN", seed="000")
good_fig.savefig("fig/plot_error_stream_lwf_adwin.pdf")

# %%
bad_f1_fig = plot_drift_detection(Path("logs"), strategy="LWF", detector="STEPD", seed="000")
bad_f1_fig.savefig("fig/plot_error_stream_stepd_lwf.pdf")


