"""Generate fig/simulation_gap.pdf: the simulation gap of open-loop detector scores.

Run with `uv run analysis/plot_simulation_gap.py` from the repository root.
One row of panels per dataset: RFMNIST, then ECovtype. Each panel shows open-loop minus
closed-loop F1, paired by seed: (a) replayed on the Oracle's streams of the
evaluation seeds, which are the CLDD-A streams of those seeds (`dd_run`), (b)
replayed on the CLDD-B streams of the other detectors (`replay_cldd_b`).
"""

import os
import sys
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from plots import barplot_simulation_gap, save_fig
from results_check import BEST

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dodo import EVALUATION_SEEDS

DETECTORS = ["ADWIN", "DDM", "PH", "SEED", "STEPD"]


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(detector_label=df["detector_label"].replace(ONE_SIDED_LABELS))
    return df[df["detector_label"].isin(DETECTORS)]


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
    closed, oracle, cldd_b = {}, {}, {}
    for dataset in ("rfmnist", "covertype"):
        root = logs_path / dataset
        ev = pd.read_csv(root / "evaluate/data_frame.csv")
        ev = ev[~ev["detector_label"].isin([*EARLY_LATE_RAW, BEST])]
        closed[dataset] = relabel(ev)
        sim = pd.read_csv(root / "dd_run/data_frame.csv")
        # dd_run stores the trial; trial i replays the stream of evaluation seed i
        sim["seed"] = sim["seed"].map(dict(enumerate(EVALUATION_SEEDS)))
        oracle[dataset] = relabel(sim)
        cldd_b[dataset] = relabel(pd.read_csv(root / "replay_cldd_b/data_frame.csv"))
    fig = barplot_simulation_gap(
        closed,
        {"(a) CLDD-A streams": oracle, "(b) CLDD-B streams": cldd_b},
    )
    save_fig(fig, "simulation_gap.pdf")


if __name__ == "__main__":
    main()
