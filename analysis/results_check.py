"""Write the dataset check tables (LaTeX): Oracle vs ADWIN vs the best detector.

Run with `uv run analysis/results_check.py`. Reads `evaluate/data_frame.csv` and the
`select_best` outputs of each dataset under `$LOGDIR/cldd`. Writes
`table/check_oracle_best.tex` (accuracy) and `table/check_oracle_best_dd.tex` (F1 and
NMDT).

ADWIN is the detector picked on RotatedFashionMNIST, so the other datasets are a held-out
test.
The best detector (`Best`) of
each strategy and boundary is picked on the tuning seeds of each dataset (see
`src/select_best.py`). Significance marks use the same tests as `results_strategy.py`.
Each detector is tested against the Oracle on its own, so the two families do not
share runs when the best detector is ADWIN.
"""

import pandas as pd

from common import ONE_SIDED_LABELS, load_best_detectors
from plots import compute_nmdt
from results_strategy import (
    BOUNDARY_AGNOSTIC,
    DOWN,
    DV,
    F1_METRIC,
    LOGS,
    TABLE_DIR,
    UP,
    format_column,
    significance_tests,
)

ORACLE = "oracle"
FIXED = "ADWIN"
BEST = "BEST"
DATASETS = {
    "rfmnist": ("RotatedFashionMNIST", LOGS),
}
BOUNDARY_ORDER = ["abrupt", "gradual", "slow"]
NO_ARROW = r"$\ \;$"


def select_runs(df_ev: pd.DataFrame, df_best: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split runs into Oracle, fixed (ADWIN) and best detector, with a `best` column.

    The main dataset ran every detector live, so its best runs are the rows of the
    picked detector. Check datasets ran the picked detector as its own `BEST` runs.
    """
    df = df_ev[~df_ev["strategy"].isin(BOUNDARY_AGNOSTIC)]
    df = df.merge(df_best, on=["boundary", "strategy"], how="left")
    has_best_runs = (df["detector_label"] == BEST).any()
    best_label = df["detector_label"] == (BEST if has_best_runs else df["best"])
    return {
        ORACLE: df[df["detector_label"] == ORACLE],
        FIXED: df[df["detector_label"] == FIXED],
        BEST: df[best_label].assign(detector_label=BEST),
    }


def arrows(runs: dict[str, pd.DataFrame], detector: str) -> dict[tuple, str]:
    """Map (boundary, strategy) to the significance arrow of `detector` vs Oracle."""
    better, worse = significance_tests(pd.concat([runs[ORACLE], runs[detector]]))
    out = {}
    for key, _ in runs[detector].groupby(["boundary", "strategy"]):
        full = (*key, detector)
        out[key] = UP if full in better else DOWN if full in worse else NO_ARROW
    return out


def mean_std(df: pd.DataFrame, col: str, scale: float) -> pd.DataFrame:
    g = df.groupby(["boundary", "strategy"])[col]
    return pd.DataFrame({"mean": g.mean() * scale, "std": g.std() * scale})


def formatted(df: pd.DataFrame, col: str, scale: float, ndigits: int) -> pd.Series:
    return format_column(mean_std(df, col, scale), ndigits=ndigits)


def dataset_columns(runs: dict[str, pd.DataFrame], baseline: pd.DataFrame, name: str):
    """Return (accuracy columns, detection columns) for one dataset.

    :param baseline: Oracle runs of the detector-agnostic strategies (FT). They get an
        accuracy row with no detector columns, as a reference.
    """
    fixed_arrow = arrows(runs, FIXED)
    best_arrow = arrows(runs, BEST)
    picked = runs[BEST].groupby(["boundary", "strategy"])["best"].first()

    acc = pd.DataFrame(
        {
            (name, "Oracle"): formatted(runs[ORACLE], DV, 100, 1),
            (name, FIXED): formatted(runs[FIXED], DV, 100, 1),
            (name, "Best"): formatted(runs[BEST], DV, 100, 1),
            (name, "Det."): picked.map(lambda d: ONE_SIDED_LABELS.get(d, d)),
        }
    )
    acc[(name, FIXED)] = acc.index.map(fixed_arrow) + acc[(name, FIXED)]
    acc[(name, "Best")] = acc.index.map(best_arrow) + acc[(name, "Best")]
    base = pd.DataFrame({(name, "Oracle"): formatted(baseline, DV, 100, 1)})
    acc = pd.concat([acc, base]).fillna("")

    dd = pd.DataFrame(
        {
            (name, f"{FIXED} F1 {UP}"): formatted(runs[FIXED], F1_METRIC, 100, 1),
            (name, f"{FIXED} NMDT {DOWN}"): formatted(runs[FIXED], "nmdt", 1, 2),
            (name, f"Best F1 {UP}"): formatted(runs[BEST], F1_METRIC, 100, 1),
            (name, f"Best NMDT {DOWN}"): formatted(runs[BEST], "nmdt", 1, 2),
        }
    )
    return acc, dd


def latex_table(df: pd.DataFrame) -> str:
    df = df.copy()
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Boundaries in order, and the detector-agnostic baseline last in each block
    strategies = sorted(
        df.index.get_level_values(1).unique(), key=lambda s: (s in BOUNDARY_AGNOSTIC, s)
    )
    df = df.reindex(
        pd.MultiIndex.from_product([BOUNDARY_ORDER, strategies]), fill_value=""
    )
    df.index.names = ["Boundary", "Strategy"]
    table = df.to_latex(escape=False, sparsify=True, multicolumn_format="c")
    # Same rules as results_strategy.py: midrule between boundaries, none at the end
    n_cols = df.index.nlevels + len(df.columns)
    table = table.replace(f"\\cline{{1-{n_cols}}}\n\\bottomrule", "\\bottomrule")
    table = table.replace(f"\\cline{{1-{n_cols}}}", "\\midrule")
    return table


if __name__ == "__main__":
    acc_parts, dd_parts = [], []
    for name, root in DATASETS.values():
        df_ev = pd.read_csv(root / "evaluate/data_frame.csv").pipe(compute_nmdt)
        runs = select_runs(df_ev, load_best_detectors(root))
        baseline = df_ev[
            df_ev["strategy"].isin(BOUNDARY_AGNOSTIC) & (df_ev["detector_label"] == ORACLE)
        ]
        acc, dd = dataset_columns(runs, baseline, name)
        print(f"== {name}")
        print(acc.to_string())
        acc_parts.append(acc)
        dd_parts.append(dd)

    for suffix, parts in [("", acc_parts), ("_dd", dd_parts)]:
        path = TABLE_DIR / f"check_oracle_best{suffix}.tex"
        print(f"Saving {path}")
        path.write_text(latex_table(pd.concat(parts, axis=1)))
