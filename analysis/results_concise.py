"""Write concise candidate result tables (LaTeX) for both datasets.

Run with `uv run analysis/results_concise.py`. Reads
`$LOGDIR/cldd/{rfmnist,covertype}/evaluate/data_frame.csv`. Writes
`table/options/option_<name>.tex`; each holds both datasets.

Every cell is mean and standard deviation (%) over the evaluation seeds. The evaluation
seeds are shared by all strategies, detectors and boundaries, so a cell that averages
over strategies or boundaries first averages each seed, then takes the standard
deviation across seeds. No table picks a winner after seeing the evaluation runs. Among
the six real detectors (ADWIN, ADWIN*, DDM, PH, SEED, STEPD), the best mean of each
column is bold and the second best is underlined. The baselines, FT and the extra rows
are not marked. `BEST` is the detector picked on the tuning seeds
(see `src/select_best.py`) and is only shown as a labelled extra row. On abrupt drift
early, mid and late are equivalent, so the early and late cells are blank there.

In the `detectors_<dataset>.tex` tables, an arrow after a detector's accuracy marks a
significant difference from the mid Oracle, a dagger a cell that is not significantly
above FT and a double dagger one significantly below FT. Each is a two-sided paired t-test
over the seeds of the strategy-averaged accuracy, Holm-corrected over the detector cells
of the table, separately for the Oracle and FT (see `results_significance.py`).
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats as sps

from common import (
    ALPHA,
    EARLY_LATE_RAW,
    MID_LABEL,
    ONE_SIDED_LABELS,
    ORACLE_RAW,
    holm,
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
DATASET_NAMES = {"rfmnist": "RFMNIST", "covertype": "ECovtype"}
# name: (header, column, scale)
METRIC = dict(zip(["acc", "f1", "bwt", "ttt"], METRICS))
# NMDT: mean detection delay over the half-width of the detection window (1 for a run
# without a detected drift), as in `plots.compute_nmdt`
METRIC["nmdt"] = ("NMDT", "nmdt", 1)
DIGITS = {"nmdt": 2}
"""Decimals of a metric, 1 if not listed."""
UNRANKED = {"nmdt"}
"""Metrics with no best value (earlier detection is not always better), so not marked."""
TINY = r"{{\tiny{}$\pm${:.{}f}}}"  # \tiny is a switch, so the braces scope it


def tiny_std(std: float, digits: int = 1) -> str:
    """The standard deviation, padded to two digits so means align on the decimal point."""
    return TINY.format(r"\phantom{0}" if round(std, digits) < 10 else "", std, digits)


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
    df["nmdt"] = (df["dd.mdt"] / df["dd.max_delay"]).fillna(1.0)
    # The baselines have no drift detector to score
    baselines = df["detector_label"].isin([MID_LABEL, *EARLY_LATE_RAW])
    df.loc[baselines, ["dd.my_f1", "nmdt"]] = np.nan
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
    # On abrupt drift early, mid and late are equivalent: only mid is shown
    if boundaries == ["abrupt"] and detector in EARLY_LATE_RAW:
        return np.nan, np.nan
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


def render(
    rows,
    cols,
    data,
    label_heads: list[str],
    marks=None,
    underline: bool = True,
    group_heads=None,
    space_after=(),
) -> str:
    """`rows`: (label cells, strategies, detector, group); `cols`: (header levels,
    dataset, boundaries, metric). `marks` maps (row, column) to text after the cell.
    With `underline` the second best of each column is underlined. `group_heads` maps a
    group to a heading row above it; its rows are then indented. Columns whose metric is in
    `space_after` get extra space on their right, for marks that would touch the next
    column."""
    group_heads = group_heads or {}
    marks = marks or {}
    stats = [
        [cell(data[ds], bds, strats, det, m) for _, ds, bds, m in cols]
        for _, strats, det, _ in rows
    ]
    means = np.array([[s[0] for s in r] for r in stats])
    # Rank only the real detectors, within each group of rows and each column: the best
    # mean is bold and the second best is underlined. Ties share a rank. The baselines,
    # FT and the extra rows are never marked.
    groups = [g for _, _, _, g in rows]
    rank = np.full(means.shape, np.inf)
    for g in set(groups):
        idx = [
            i for i, (_, _, det, x) in enumerate(rows) if x == g and det in DETECTORS
        ]
        for c in range(means.shape[1]):
            if cols[c][3] in UNRANKED:
                continue
            values = np.round(means[idx, c], 1)
            distinct = sorted({v for v in values if not np.isnan(v)}, reverse=True)
            for i, v in zip(idx, values):
                if not np.isnan(v):
                    rank[i, c] = distinct.index(v)
    n_label = len(label_heads)
    lines = [
        r"\begin{tabular}{"
        + "l" * n_label
        + "".join(
            r"r@{\hspace{\dimexpr2\tabcolsep+4pt\relax}}"
            if m in space_after and c < len(cols) - 1
            else "r"
            for c, (_, _, _, m) in enumerate(cols)
        )
        + "}",
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
        if group in group_heads and group != previous:
            width = n_label + len(cols)
            lines.append(rf"\multicolumn{{{width}}}{{l}}{{{group_heads[group]}}} \\")
        if group in group_heads:
            labels = [r"\hspace{0.8em}" + labels[0], *labels[1:]]
        previous = group
        text = []
        for c in range(len(cols)):
            mean, std = stats[r][c]
            if np.isnan(mean):
                text.append("")
                continue
            digits = DIGITS.get(cols[c][3], 1)
            body = f"{mean:.{digits}f}" + tiny_std(std, digits)
            if rank[r, c] == 0:
                body = rf"\textbf{{{body}}}"
            elif underline and rank[r, c] == 1:
                body = rf"\underline{{{body}}}"
            text.append(body + marks.get((r, c), ""))
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


def significance(rows, cols, data, strategies, metric="acc") -> dict:
    """Test each detector `metric` cell against a reference, Holm-corrected over the cells
    of that metric.

    `strategies` picks the reference: None compares with the mid Oracle on the same
    strategies, ["FT"] with FT. Returns (row, column) -> (mean difference, significant).
    """
    tests = []
    for r, (_, strats, det, _) in enumerate(rows):
        if det not in DETECTORS:
            continue
        for c, (_, ds, bds, m) in enumerate(cols):
            if m != metric:
                continue
            a = per_seed(data[ds], bds, strats, det, metric)
            b = per_seed(data[ds], bds, strategies or strats, MID_LABEL, metric)
            diff = (a - b).dropna()
            tests.append((r, c, diff.mean(), sps.ttest_1samp(diff, 0.0).pvalue))
    adjusted = holm([p for *_, p in tests])
    return {(r, c): (mean, p < ALPHA) for (r, c, mean, _), p in zip(tests, adjusted)}


def detector_marks(rows, cols, data, metrics=("acc", "bwt")) -> dict:
    """Marks after a detector's accuracy and BWT: an arrow for a significant difference from
    the mid Oracle, a dagger when the cell is not significantly above FT (double dagger:
    below). Each metric is its own Holm family. The header arrow gives the better direction.

    The marks have zero width, so they do not shift the decimal point of their column.
    """
    oracle, ft = {}, {}
    for metric in metrics:
        oracle |= significance(rows, cols, data, None, metric)
        ft |= significance(rows, cols, data, ["FT"], metric)
    marks = {}
    for key, (mean, sig) in oracle.items():
        text = (r"\uparrow" if mean > 0 else r"\downarrow") if sig else ""
        ft_mean, ft_sig = ft[key]
        if not ft_sig:
            text += r"\dagger"
        elif ft_mean < 0:
            text += r"\ddagger"
        if text:
            marks[key] = rf"\rlap{{$^{{{text}}}$}}"
    return marks


def build_summary(data) -> dict[str, str]:
    """Option A with one table per dataset, and BWT and test-then-train accuracy next to
    CL accuracy and F1."""
    per_boundary = [(b.capitalize(), [b]) for b in BOUNDARIES]
    tables = {}
    for name in DATASETS:
        rows = detector_rows(DETECTORS)
        cols = columns(
            ["acc", "bwt", "f1", "nmdt"], per_boundary, [name], dataset_level=False
        )
        # Higher is better for these. NMDT has no better direction, so no arrow.
        cols = [
            ([*h[:-1], h[-1] + r"$\uparrow$"] if m in ("acc", "bwt", "f1") else h, ds, bds, m)
            for h, ds, bds, m in cols
        ]
        marks = detector_marks(rows, cols, data)
        tables[name] = render(
            rows,
            cols,
            data,
            ["Detector"],
            marks,
            underline=False,
            group_heads={"control": r"\textit{Oracle}"},
            space_after=("acc", "bwt"),
        )
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detectors-only",
        action="store_true",
        help="write only table/detectors_<dataset>.tex, leaving option tables untouched",
    )
    args = parser.parse_args()

    data = {name: load_runs(root) for name, root in DATASETS.items()}
    if not args.detectors_only:
        OPTION_DIR.mkdir(exist_ok=True)
        for old in OPTION_DIR.glob("option_*"):
            old.unlink()
        for name, text in build(data).items():
            path = OPTION_DIR / f"option_{name}.tex"
            print(f"Saving {path}")
            path.write_text(text)
    for name, text in build_summary(data).items():
        path = TABLE_DIR / f"detectors_{name}.tex"
        print(f"Saving {path}")
        path.write_text(text)
