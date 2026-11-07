"""Write the full result table (LaTeX) of each dataset: every strategy and detector.

Run with `uv run analysis/results_full.py`. Reads
`$LOGDIR/cldd/{rfmnist,covertype}/evaluate/data_frame.csv`. Writes
`table/full_rfmnist.tex` and `table/full_covertype.tex`.

One row per strategy and detector (FT has no detector). One group of columns per
boundary: CL accuracy, drift detection F1, backward transfer and test-then-train
accuracy, as mean and standard deviation over the evaluation seeds. The best mean of
each column is bold. A detector that did not run on a dataset is left blank.
"""

import pandas as pd
from common import ONE_SIDED_LABELS
from results_strategy import LOGS, TABLE_DIR, format_column

STRATEGIES = ["EWC", "LWF", "MAS", "RWalk", "SI"]
DETECTORS = ["ADWIN", "ADWIN*", "DDM", "PH", "SEED", "STEPD", "Oracle"]
BOUNDARIES = ["abrupt", "gradual", "slow"]
# (header, column, scale)
METRICS = [
    ("Acc.", "ocl.accuracy_seen_avg", 100),
    ("F1", "dd.my_f1", 100),
    ("BWT", "ocl.backward_transfer", 100),
    (r"Acc$_{ttt}$", "ttt.accuracy", 1),
]
DATASETS = {"rfmnist": LOGS, "covertype": LOGS.parent / "covertype"}


def load(root) -> pd.DataFrame:
    df = pd.read_csv(root / "evaluate/data_frame.csv")
    df = df.drop(columns="dataset", errors="ignore")
    df["detector_label"] = (
        df["detector_label"]
        .replace(ONE_SIDED_LABELS)
        .replace({"ADWIN_JOINT": "ADWIN*", "oracle": "Oracle"})
    )
    cols = [c for _, c, _ in METRICS]
    # The oracle has no drift detector to score
    df.loc[df["detector_label"] == "Oracle", "dd.my_f1"] = float("nan")
    return df.groupby(["boundary", "strategy", "detector_label"])[cols].agg(
        ["mean", "std"]
    )


def row_keys() -> list[tuple[str, str]]:
    return [(s, d) for s in STRATEGIES for d in DETECTORS] + [("FT", "Oracle")]


def format_cells(stats: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Formatted cells of every row, in the order of `BOUNDARIES` x `METRICS`."""
    keys = row_keys()
    cells: dict[tuple[str, str], list[str]] = {k: [] for k in keys}
    for boundary in BOUNDARIES:
        for _, col, scale in METRICS:
            values = pd.DataFrame(
                {
                    "mean": [
                        _get(stats, boundary, k, col, "mean") * scale for k in keys
                    ],
                    "std": [_get(stats, boundary, k, col, "std") * scale for k in keys],
                },
                index=pd.MultiIndex.from_tuples(keys),
            )
            formatted = format_column(values, max_style="textbf", ndigits=1, nan_str="")
            for k, text in zip(keys, formatted):
                cells[k].append(text)
    return cells


def _get(stats: pd.DataFrame, boundary: str, key, col: str, agg: str) -> float:
    try:
        return float(stats.loc[(boundary, key[0], key[1]), (col, agg)])
    except KeyError:
        return float("nan")


def latex_table(cells: dict[tuple[str, str], list[str]]) -> str:
    n_cols = len(METRICS)
    total = 2 + n_cols * len(BOUNDARIES)
    lines = [r"\begin{tabular}{ll" + "r" * (total - 2) + "}", r"\toprule"]
    groups = " & ".join(
        rf"\multicolumn{{{n_cols}}}{{c}}{{{b.capitalize()}}}" for b in BOUNDARIES
    )
    lines.append(f" & & {groups} \\\\")
    rules = "".join(
        rf"\cmidrule(lr){{{3 + i * n_cols}-{2 + (i + 1) * n_cols}}}"
        for i in range(len(BOUNDARIES))
    )
    lines.append(rules)
    heads = " & ".join(h for _ in BOUNDARIES for h, _, _ in METRICS)
    lines.append(f"Strategy & Detector & {heads} \\\\")
    lines.append(r"\midrule")
    for s in STRATEGIES:
        for i, d in enumerate(DETECTORS):
            label = rf"\multirow[t]{{{len(DETECTORS)}}}{{*}}{{{s}}}" if i == 0 else ""
            lines.append(f"{label} & {d} & {' & '.join(cells[(s, d)])} \\\\")
        lines.append(r"\midrule")
    lines.append(f"FT & & {' & '.join(cells[('FT', 'Oracle')])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    for name, root in DATASETS.items():
        path = TABLE_DIR / f"full_{name}.tex"
        print(f"Saving {path}")
        path.write_text(latex_table(format_cells(load(root))))
