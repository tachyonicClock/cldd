"""Write concise candidate result tables (LaTeX) for both datasets.

Run with `uv run analysis/results_concise.py`. Reads
`$LOGDIR/cldd/{rfmnist,covertype}/evaluate/data_frame.csv`. Writes
`table/options/option_<name>.tex`; each holds both datasets.

Every cell is mean and standard deviation (%) over the evaluation seeds. The evaluation
seeds are shared by all strategies, detectors and boundaries, so a cell that averages
over strategies or boundaries first averages each seed, then takes the standard
deviation across seeds. No table picks a winner after seeing the evaluation runs. The
best mean of each column is bold. `BEST` is the detector picked on the tuning seeds
(see `src/select_best.py`) and is only shown as a labelled extra row.
"""

import numpy as np
import pandas as pd
from common import (
    EARLY_LATE_RAW,
    MID_LABEL,
    ONE_SIDED_LABELS,
    ORACLE_RAW,
    load_best_detectors,
)
from results_check import BEST, select_runs
from results_full import BASELINES, DATASETS, METRICS
from results_strategy import TABLE_DIR

OPTION_DIR = TABLE_DIR / "options"
BOUNDARIES = ["abrupt", "gradual", "slow"]
STRATEGIES = ["EWC", "LWF", "MAS", "RWalk", "SI"]
DETECTORS = ["ADWIN", "ADWIN*", "DDM", "PH", "SEED", "STEPD"]
MEDIAN = "MEDIAN"
BEST_LABEL = r"BEST$^\dagger$"
DATASET_NAMES = {"rfmnist": "RFMNIST", "covertype": "Covertype"}
# name: (header, column, scale)
METRIC = dict(zip(["acc", "f1", "bwt", "ttt"], METRICS))
TINY = r"\tiny{{$\pm${:.1f}}}"


def load_runs(root) -> pd.DataFrame:
    """Every evaluation run, with the runs of the BEST detector under `BEST`."""
    df_ev = pd.read_csv(root / "evaluate/data_frame.csv")
    best = select_runs(df_ev, load_best_detectors(root))[BEST]
    df = pd.concat([df_ev[df_ev["detector_label"] != BEST], best], ignore_index=True)
    df["detector_label"] = (
        df["detector_label"]
        .replace(ONE_SIDED_LABELS)
        .replace({"ADWIN_JOINT": "ADWIN*", ORACLE_RAW: MID_LABEL})
    )
    # The baselines have no drift detector to score
    df.loc[df["detector_label"].isin([MID_LABEL, *EARLY_LATE_RAW]), "dd.my_f1"] = np.nan
    return df


def per_seed(df, boundaries, strategies, detector, metric) -> pd.Series:
    """The metric of each seed, averaged over boundaries and strategies."""
    _, col, scale = METRIC[metric]
    if strategies == ["FT"]:
        detector = MID_LABEL
    d = df[df["boundary"].isin(boundaries) & df["strategy"].isin(strategies)]
    if detector == MEDIAN:
        d = d[d["detector_label"].isin(DETECTORS)]
        # Median over the detectors that ran, for each boundary, strategy and seed
        d = d.groupby(["boundary", "strategy", "seed"])[col].median().reset_index()
    else:
        d = d[d["detector_label"] == detector]
    return d.groupby("seed")[col].mean() * scale


def cell(df, boundaries, strategies, detector, metric) -> tuple[float, float]:
    values = per_seed(df, boundaries, strategies, detector, metric).dropna()
    if values.empty:
        return np.nan, np.nan
    return float(values.mean()), float(values.std())


def spans(labels: list[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for label in labels:
        if out and out[-1][0] == label:
            out[-1] = (label, out[-1][1] + 1)
        else:
            out.append((label, 1))
    return out


def render(rows, cols, data, label_heads: list[str]) -> str:
    """`rows`: (label cells, strategies, detector, group); `cols`: (header levels,
    dataset, boundaries, metric)."""
    stats = [
        [cell(data[ds], bds, strats, det, m) for _, ds, bds, m in cols]
        for _, strats, det, _ in rows
    ]
    means = np.array([[s[0] for s in r] for r in stats])
    best = np.nanmax(means, axis=0)
    n_label = len(label_heads)
    lines = [
        r"\begin{tabular}{" + "l" * n_label + "r" * len(cols) + "}",
        r"\toprule",
    ]
    n_levels = len(cols[0][0])
    for level in range(n_levels):
        labels = [c[0][level] for c in cols]
        if level == n_levels - 1:
            lines.append(" & ".join(label_heads) + " & " + " & ".join(labels) + r" \\")
            continue
        heads = [rf"\multicolumn{{{n}}}{{c}}{{{label}}}" for label, n in spans(labels)]
        lines.append(" & " * n_label + " & ".join(heads) + r" \\")
        start, rules = 1 + n_label, ""
        for _, n in spans(labels):
            rules += rf"\cmidrule(lr){{{start}-{start + n - 1}}}"
            start += n
        lines.append(rules)
    lines.append(r"\midrule")
    previous = None
    for r, (labels, _, _, group) in enumerate(rows):
        if previous is not None and group != previous:
            lines.append(r"\midrule")
        previous = group
        text = []
        for c in range(len(cols)):
            mean, std = stats[r][c]
            if np.isnan(mean):
                text.append("")
                continue
            body = f"{mean:.1f}" + TINY.format(std)
            if round(mean, 1) == round(best[c], 1):
                body = rf"\textbf{{{body}}}"
            text.append(body)
        lines.append(" & ".join([*labels, *text]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def strategy_rows(detectors):
    """One row per strategy and detector, then FT."""
    rows = []
    for s in STRATEGIES:
        for i, d in enumerate(detectors):
            label = rf"\multirow[t]{{{len(detectors)}}}{{*}}{{{s}}}" if i == 0 else ""
            rows.append(([label, d], [s], d, s))
    rows.append((["FT", ""], ["FT"], MID_LABEL, "FT"))
    return rows


def detector_rows(detectors, extras=()):
    """One row per detector, averaged over strategies, then the baselines, extras and FT."""
    rows = [([d], STRATEGIES, d, "detectors") for d in detectors]
    rows += [([b], STRATEGIES, b, "control") for b in BASELINES]
    rows += [([label], STRATEGIES, det, "extra") for label, det in extras]
    rows.append((["FT"], ["FT"], MID_LABEL, "FT"))
    return rows


def columns(metrics, boundary_groups, datasets, dataset_level=True):
    """Headers: dataset > boundary > metric (no boundary level for the average)."""
    cols = []
    for ds in datasets:
        for name, bds in boundary_groups:
            for m in metrics:
                heads = [DATASET_NAMES[ds], name, METRIC[m][0]]
                if name == "Avg.":
                    heads.remove(name)
                if not dataset_level:
                    heads.pop(0)
                cols.append((heads, ds, bds, m))
    return cols


def build(data) -> dict[str, str]:
    both = list(DATASETS)
    per_boundary = [(b.capitalize(), [b]) for b in BOUNDARIES]
    average = [("Avg.", BOUNDARIES)]
    spread = [("Median detector", MEDIAN), (BEST_LABEL, BEST)]
    return {
        # A: every detector, averaged over strategies, per boundary
        "A_detectors_per_boundary": render(
            detector_rows(DETECTORS),
            columns(["acc", "f1"], per_boundary, both),
            data,
            ["Detector"],
        ),
        # B: every detector, averaged over strategies and boundaries, four metrics
        "B_detectors_average": render(
            detector_rows(DETECTORS),
            columns(["acc", "f1", "bwt", "ttt"], average, both),
            data,
            ["Detector"],
        ),
        # C: per strategy with the fixed detectors only, so nothing is selected
        "C_strategy_fixed_detectors": render(
            strategy_rows(["ADWIN", "ADWIN*", *BASELINES]),
            columns(["acc", "f1"], per_boundary, both),
            data,
            ["Strategy", "Detector"],
        ),
        # D: the spread over detectors: median next to the selected detector
        "D_median_vs_selected": render(
            detector_rows(["ADWIN", "ADWIN*"], extras=spread),
            columns(["acc", "f1", "bwt"], per_boundary, both),
            data,
            ["Detector"],
        ),
    }


def build_summary(data) -> dict[str, str]:
    """Option A with one table per dataset, and BWT next to CL accuracy and F1."""
    per_boundary = [(b.capitalize(), [b]) for b in BOUNDARIES]
    return {
        name: render(
            detector_rows(DETECTORS),
            columns(["acc", "f1", "bwt"], per_boundary, [name], dataset_level=False),
            data,
            ["Detector"],
        )
        for name in DATASETS
    }


if __name__ == "__main__":
    OPTION_DIR.mkdir(exist_ok=True)
    for old in OPTION_DIR.glob("option_*"):
        old.unlink()
    data = {name: load_runs(root) for name, root in DATASETS.items()}
    for name, text in build(data).items():
        path = OPTION_DIR / f"option_{name}.tex"
        print(f"Saving {path}")
        path.write_text(text)
    for name, text in build_summary(data).items():
        path = TABLE_DIR / f"detectors_{name}.tex"
        print(f"Saving {path}")
        path.write_text(text)
