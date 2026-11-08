"""Generate only fig/contour_acc.pdf from the RFMNIST evaluation CSV under $LOGDIR.

Run with `uv run analysis/plot_contour_acc.py` from the repository root.
"""

import os
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from plots import contour_acc_nmdt_f1_boundary, save_fig


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd" / "rfmnist"
    df = pd.read_csv(logs_path / "evaluate/data_frame.csv")
    df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
    df = df[~df["detector_label"].isin(EARLY_LATE_RAW)]
    save_fig(contour_acc_nmdt_f1_boundary(df), "contour_acc.pdf")


if __name__ == "__main__":
    main()
