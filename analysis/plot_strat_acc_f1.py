"""Generate fig/strat_acc_f1.pdf from the RFMNIST evaluation CSV under $LOGDIR.

Run with `uv run analysis/plot_strat_acc_f1.py` from the repository root.
"""

import os
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from plots import save_fig, scatterplot_acc_f1_strategy_boundary


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd" / "rfmnist"
    df = pd.read_csv(logs_path / "evaluate/data_frame.csv")
    df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
    df = df[~df["detector_label"].isin(EARLY_LATE_RAW)]
    fig, r_squared = scatterplot_acc_f1_strategy_boundary(df)
    save_fig(fig, "strat_acc_f1.pdf")
    print(
        f"strat_acc_f1: R^2={ {boundary: round(float(value), 3) for boundary, value in r_squared.items()} }"
    )


if __name__ == "__main__":
    main()
