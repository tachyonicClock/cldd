"""Generate fig/hm_f1.pdf from the evaluation CSVs under $LOGDIR.

Run with `uv run analysis/plot_hm_f1.py` from the repository root.
"""

import os
from pathlib import Path

import pandas as pd
from common import ONE_SIDED_LABELS, TEXTWIDTH
from plots import heatmap_f1_strategy_detector_boundary, save_fig


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
    heat_data = {}
    for dataset in ("rfmnist", "covertype"):
        df = pd.read_csv(logs_path / dataset / "evaluate/data_frame.csv")
        df["detector_label"] = df["detector_label"].replace(ONE_SIDED_LABELS)
        heat_data[dataset] = df
    save_fig(heatmap_f1_strategy_detector_boundary(heat_data, width=TEXTWIDTH, font=8, aspect=0.7), "hm_f1.pdf")


if __name__ == "__main__":
    main()
