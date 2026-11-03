"""Write the dataset check table (LaTeX): Oracle vs the best detector on each dataset.

Run with `uv run analysis/results_check.py`. Reads `evaluate/data_frame.csv` and the
`select_best` outputs of each dataset under `$LOGDIR/cldd`. Writes
`table/check_oracle_best.tex`.

The best detector of each strategy and boundary is picked on the tuning seeds (see
`src/select_best.py`). Significance marks use the same tests as
`results_strategy.py`, run per dataset on the Oracle and best detector runs only.
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

BEST = "BEST"
DATASETS = {"rfmnist": ("RotatedFashionMNIST", LOGS), "dcifar100": ("DomainCIFAR100", LOGS / "dcifar100")}
BOUNDARY_ORDER = ["abrupt", "gradual", "slow"]


def oracle_and_best(df_ev: pd.DataFrame, df_best: pd.DataFrame) -> pd.DataFrame:
    """Keep Oracle runs and best-detector runs, the latter labelled `BEST`.

    The main dataset ran every detector live, so its best runs are the rows of the
    picked detector. Check datasets ran only the picked detector, already labelled.
    """
    df_ev = df_ev[~df_ev["strategy"].isin(BOUNDARY_AGNOSTIC)]
    df = df_ev.merge(df_best, on=["boundary", "strategy"], how="left")
    is_best = (df["detector_label"] == df["best"]) | (df["detector_label"] == BEST)
    df = df[is_best | (df["detector_label"] == "oracle")].copy()
    df.loc[is_best, "detector_label"] = BEST
    return df


def summarise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    better, worse = significance_tests(df)
    agg = df.groupby(["boundary", "strategy", "detector_label"]).aggregate(
        {DV: ["mean", "std"], F1_METRIC: ["mean", "std"], "nmdt": ["mean", "std"]}
    )
    rows = []
    for (boundary, strategy), group in agg.groupby(level=[0, 1]):
        oracle = group.xs("oracle", level=2).iloc[0]
        best = group.xs(BEST, level=2).iloc[0]
        key = (boundary, strategy, BEST)
        arrow = UP if key in better else DOWN if key in worse else r"$\ \;$"
        picked = df.loc[
            (df["boundary"] == boundary) & (df["strategy"] == strategy), "best"
        ].iloc[0]
        rows.append(
            {
                "boundary": boundary,
                "strategy": strategy,
                (name, "Det."): ONE_SIDED_LABELS.get(picked, picked),
                (name, "Oracle"): (oracle[(DV, "mean")] * 100, oracle[(DV, "std")] * 100),
                (name, "Best"): (best[(DV, "mean")] * 100, best[(DV, "std")] * 100, arrow),
                (name, f"F1 {UP}"): (
                    best[(F1_METRIC, "mean")] * 100,
                    best[(F1_METRIC, "std")] * 100,
                ),
                (name, f"NMDT {DOWN}"): (best[("nmdt", "mean")], best[("nmdt", "std")]),
            }
        )
    return pd.DataFrame(rows).set_index(["boundary", "strategy"])


def cell(value, ndigits: int) -> str:
    mean, std, *arrow = value
    text = format_column(
        pd.DataFrame({"mean": [mean], "std": [std]}), ndigits=ndigits
    ).iloc[0]
    return (arrow[0] if arrow else "") + text


def latex_table(df: pd.DataFrame) -> str:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        name, metric = col
        if metric == "Det.":
            out[col] = df[col]
        else:
            ndigits = 2 if metric.startswith("NMDT") else 1
            out[col] = df[col].map(lambda v: cell(v, ndigits))
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out = out.reindex(BOUNDARY_ORDER, level=0)
    out.index.names = ["Boundary", "Strategy"]
    table = out.to_latex(escape=False, sparsify=True, multicolumn_format="c")
    # Same rules as results_strategy.py: midrule between boundaries, none at the end
    n_cols = out.index.nlevels + len(out.columns)
    table = table.replace(f"\\cline{{1-{n_cols}}}\n\\bottomrule", "\\bottomrule")
    table = table.replace(f"\\cline{{1-{n_cols}}}", "\\midrule")
    return table


if __name__ == "__main__":
    parts = []
    for key, (name, root) in DATASETS.items():
        df_ev = pd.read_csv(root / "evaluate/data_frame.csv").pipe(compute_nmdt)
        df = oracle_and_best(df_ev, load_best_detectors(root))
        print(f"== {name}: picked detectors")
        print(df.groupby(["boundary", "strategy"])["best"].first().unstack(0))
        parts.append(summarise(df, name))
    table = latex_table(pd.concat(parts, axis=1))
    path = TABLE_DIR / "check_oracle_best.tex"
    print(f"Saving {path}")
    path.write_text(table)
