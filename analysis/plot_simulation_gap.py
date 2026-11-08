"""Generate fig/simulation_gap.pdf comparing active and replayed detector F1.

Run with `uv run analysis/plot_simulation_gap.py` from the repository root.
"""

import os
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from plots import barplot_f1_detector_strategy, save_fig


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd" / "rfmnist"
    df_eval = pd.read_csv(logs_path / "evaluate/data_frame.csv")
    df_sim = pd.read_csv(logs_path / "dd_run/data_frame.csv")
    for df in (df_eval, df_sim):
        df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
    df_eval = df_eval[~df_eval["detector_label"].isin(EARLY_LATE_RAW)]
    save_fig(barplot_f1_detector_strategy(df_eval, df_sim), "simulation_gap.pdf")


if __name__ == "__main__":
    main()
