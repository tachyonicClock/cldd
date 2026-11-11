"""Generate fig/strat_acc_f1.pdf from the evaluation CSVs under $LOGDIR.

Run with `uv run analysis/plot_strat_acc_f1.py` from the repository root.
One row of panels per dataset: RFMNIST, then ECovtype.
"""

import os
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from results_check import BEST
from plots import save_fig, scatterplot_acc_f1_strategy_boundary


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
    data = {}
    for dataset in ("rfmnist", "covertype"):
        df = pd.read_csv(logs_path / dataset / "evaluate/data_frame.csv")
        df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
        # BEST repeats runs of the detector picked per strategy, so it is left out
        data[dataset] = df[~df["detector_label"].isin([*EARLY_LATE_RAW, BEST])]
    fig, r_squared = scatterplot_acc_f1_strategy_boundary(data)
    save_fig(fig, "strat_acc_f1.pdf")
    print(
        f"strat_acc_f1: R^2={ {key: round(float(value), 3) for key, value in r_squared.items()} }"
    )


if __name__ == "__main__":
    main()
