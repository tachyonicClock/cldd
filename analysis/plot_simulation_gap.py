"""Generate fig/simulation_gap.pdf comparing active and replayed detector F1.

Run with `uv run analysis/plot_simulation_gap.py` from the repository root.
One row of panels per dataset: RFMNIST, then ECovtype.
"""

import os
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from plots import barplot_f1_detector_strategy, save_fig
from results_check import BEST


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
    df_eval, df_sim = {}, {}
    for dataset in ("rfmnist", "covertype"):
        ev = pd.read_csv(logs_path / dataset / "evaluate/data_frame.csv")
        sim = pd.read_csv(logs_path / dataset / "dd_run/data_frame.csv")
        for df in (ev, sim):
            df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
        # BEST repeats runs of the detector picked per strategy, so it is left out
        df_eval[dataset] = ev[~ev["detector_label"].isin([*EARLY_LATE_RAW, BEST])]
        df_sim[dataset] = sim
    save_fig(barplot_f1_detector_strategy(df_eval, df_sim), "simulation_gap.pdf")


if __name__ == "__main__":
    main()
