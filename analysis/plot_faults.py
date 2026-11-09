"""Generate fig/faults.pdf: controlled faults on the Oracle (RFMNIST).

Run with `uv run analysis/plot_faults.py` from the repository root. Reads the evaluate
and `synthetic` runs under `$LOGDIR/cldd/rfmnist`.

Each point is the accuracy gap to the Oracle (points) of the `synthetic` run with the
fault, paired by seed and averaged over the five strategies (EWC, LwF, MAS, RWalk, SI).
FT is left out: it has no detector, so there are no faults to inject. The shaded band
is the mean plus and minus one standard deviation over the seeds. The Oracle is the one of the
evaluate runs. At 0 faults the point is the `oracle_off0` run, which replays the
Oracle's alarms with the schedule detector. Panels: (a) drifts dropped, (b) random alarms added, (c) alarms shifted.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from common import (
    FONT_LABEL,
    FONT_SUBTITLE,
    FONT_TICK,
    NO_TOP_RIGHT,
    PAPER_STYLE,
)
from plots import save_fig
from results_triggers import (
    BOUNDARIES,
    LOGS,
    STRATEGIES,
    common_seeds,
    from_evaluate,
    load_synthetic,
)

# Number of seeds used (the first trials). None uses all of them.
N_SEEDS = None
# Ordered, colour-blind safe: fast to slow drift from dark to light
COLORS = dict(zip(BOUNDARIES, cm.batlow([0.05, 0.4, 0.72])))
MARKERS = {"abrupt": "o", "gradual": "s", "slow": "^"}
# (title, x label, [(x, variant)], x tick labels)
PANELS = [
    (
        "(a) Missed drifts",
        "Drifts dropped (of 4)",
        [(0, "oracle_off0")] + [(n, f"oracle_miss{n}") for n in (1, 2, 3, 4)],
        None,
    ),
    (
        "(b) False alarms",
        "Random alarms added",
        [(0, "oracle_off0")] + [(n, f"oracle_extra{n}") for n in (3, 10, 20)],
        None,
    ),
    (
        "(c) Alarm timing",
        "Shift (1000 inst.)",
        [(-18, "oracle_off-18k"), (-6, "oracle_off-6k"), (0, "oracle_off0")]
        + [(6, "oracle_off+6k"), (18, "oracle_off+18k")],
        None,
    ),
]


def gaps(variant: str, oracle, boundary: str) -> np.ndarray:
    """Gap to the Oracle (points) of each seed, averaged over the strategies."""
    runs = load_synthetic(variant)
    seeds = common_seeds(runs)
    first = list(runs[(boundary, STRATEGIES[0])].index)  # trial order
    seeds = [s for s in first if s in seeds][:N_SEEDS]
    per_strategy = [
        100 * (runs[(boundary, st)].loc[seeds] - oracle[(boundary, st)].loc[seeds])
        for st in STRATEGIES
    ]
    return np.mean(per_strategy, axis=0)


def main() -> None:
    plt.style.use(PAPER_STYLE)
    plt.rcParams.update(NO_TOP_RIGHT)
    df = pd.read_csv(LOGS / "evaluate/data_frame.csv")
    oracle = from_evaluate(df, "oracle")

    fig, axes = plt.subplots(
        1, 3, figsize=(252 / 72, 2.3), sharey=True, constrained_layout=True
    )
    for ax, (title, xlabel, points, _) in zip(axes, PANELS):
        xs = [x for x, _ in points]
        for boundary in BOUNDARIES:
            g = [gaps(v, oracle, boundary) for _, v in points]
            mean = np.array([x.mean() for x in g])
            std = np.array([x.std(ddof=1) for x in g])
            ax.fill_between(
                xs, mean - std, mean + std, color=COLORS[boundary], alpha=0.2, lw=0
            )
            ax.plot(
                xs,
                mean,
                color=COLORS[boundary],
                marker=MARKERS[boundary],
                markersize=3.5,
                linewidth=1.0,
                label=boundary,
            )
        ax.set_xticks(xs)
        ax.set_title(title, fontsize=FONT_SUBTITLE)
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.minorticks_off()
        ax.set_yticks([0, -5, -10, -15])
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Acc gap to Oracle", fontsize=FONT_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        fontsize=FONT_TICK,
        frameon=False,
    )
    save_fig(fig, "faults.pdf")


if __name__ == "__main__":
    main()
