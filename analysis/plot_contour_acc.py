"""Generate only fig/contour_acc.pdf from the evaluation CSVs under $LOGDIR.

Run with `uv run analysis/plot_contour_acc.py` from the repository root.
One row of panels per dataset: RFMNIST, then ECovtype.
"""

import os
from pathlib import Path

import pandas as pd
from common import EARLY_LATE_RAW, ONE_SIDED_LABELS
from results_check import BEST
from plots import contour_acc_nmdt_f1_boundary, save_fig


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
    data = {}
    for dataset in ("rfmnist", "covertype"):
        df = pd.read_csv(logs_path / dataset / "evaluate/data_frame.csv")
        df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
        # BEST repeats runs of the detector picked per strategy, so it is left out
        data[dataset] = df[~df["detector_label"].isin([*EARLY_LATE_RAW, BEST])]
    save_fig(contour_acc_nmdt_f1_boundary(data), "contour_acc.pdf")


if __name__ == "__main__":
    main()
