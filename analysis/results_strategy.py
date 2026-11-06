"""Write the strategy result tables (LaTeX) from the evaluate runs.

Run with `uv run analysis/results_strategy.py`. Reads `$LOGDIR/cldd/evaluate/data_frame.csv`.
Writes `table/strategy_<boundary>.tex` (Oracle, ADWIN and ADWIN* only) and
`table/appendix_strategy_<boundary>.tex` (all detectors).

Statistically significant deviation from the control (Oracle) in accuracy is marked with
an up or down arrow. We run Welch's ANOVA with Benjamini-Hochberg correction for each
boundary and strategy, then a Games-Howell test (p=0.05).
"""

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pingouin as pg
from common import ONE_SIDED_LABELS
from jinja2 import Template
from plots import compute_nmdt

F1_METRIC = "dd.my_f1"
SIGNIFICANCE_LEVEL = 0.05
DV = "ocl.accuracy_seen_avg"
BETWEEN = "detector_label"
BOUNDARY_AGNOSTIC = {"FT"}
AGGREGATIONS = ("mean", "std")
UP = r"$\uparrow$"
DOWN = r"$\downarrow$"

LOGS = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
TABLE_DIR = Path(__file__).resolve().parent.parent / "table"

_template = Template(
    r"{% if font_style %}\{{font_style ~ '{'}}{% endif %}"
    r"{{ '{:.{}f}'.format(mean, ndigits) }}\tiny{$\pm${{ '{:.{}f}'.format(std, ndigits) ~ '}' }}"
    r"{% if font_style %}}{% endif %}"
)

TexFontStyle = Literal["textmd", "textbf", "textup", "textit", "textsl", "textsc"]


def format_column(
    df: pd.DataFrame | pd.Series,
    mean_col="mean",
    std_col="std",
    min_style: TexFontStyle | None = None,
    max_style: TexFontStyle | None = None,
    ndigits=3,
    nan_str="N/A",
) -> pd.DataFrame | pd.Series:
    col_max = df[mean_col].max()
    col_min = df[mean_col].min()

    def _func(row: pd.Series) -> str:
        mean = float(row[mean_col])
        std = float(row[std_col])
        font_style = None

        if pd.isna(mean) or pd.isna(std):
            return nan_str

        # Compare at the precision of the formatted output
        if max_style and round(mean, ndigits) == round(col_max, ndigits):
            font_style = max_style
        elif min_style and round(mean, ndigits) == round(col_min, ndigits):
            font_style = min_style

        return _template.render(
            mean=mean, std=std, font_style=font_style, ndigits=ndigits
        )

    return df.apply(_func, axis=1)


def significance_tests(df_ev: pd.DataFrame) -> tuple[set, set]:
    """Return (better, worse): (boundary, strategy, detector) cells that differ from Oracle."""
    better: set[tuple[str, str, str]] = set()
    worse: set[tuple[str, str, str]] = set()

    anova_results = []
    for i, (keys, context) in enumerate(df_ev.groupby(["boundary", "strategy"])):
        if keys[1] == "FT":
            continue
        df_anova = pg.anova(data=context, dv=DV, between=BETWEEN)
        anova_results.append(
            {
                "index": i,
                "keys": keys,
                "context": context,
                "p_unc": df_anova["p_unc"].values[0],
            }
        )

    # Benjamini-Hochberg false discovery rate correction over all contexts
    p_vals = [res["p_unc"] for res in anova_results]
    reject, _ = pg.multicomp(p_vals, alpha=SIGNIFICANCE_LEVEL, method="fdr_bh")

    for i, res in enumerate(anova_results):
        if not reject[i]:
            print(f"{res['index']} {res['keys']} not significant after FDR correction")
            continue
        # Games-Howell already adjusts for the pairwise comparisons in this context
        df_ph = pg.pairwise_gameshowell(data=res["context"], dv=DV, between=BETWEEN)
        for _, row in df_ph.iterrows():
            if row["B"] != "oracle" or row["pval"] > SIGNIFICANCE_LEVEL:
                continue
            key = (str(res["keys"][0]), str(res["keys"][1]), str(row["A"]))
            (better if row["T"] >= 0 else worse).add(key)

    print("Significantly better than Oracle:", sorted(better))
    print("Significantly worse than Oracle:", sorted(worse))
    return better, worse


def latex_table(
    df: pd.DataFrame, boundary: str, better: set, worse: set
) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(inplace=True)
    # FT does not use a detector, so put it at the end
    agnostic = df["strategy"].isin(BOUNDARY_AGNOSTIC)
    df = pd.concat([df[~agnostic], df[agnostic]], ignore_index=True)
    # Keep the raw detector labels for the significance lookup
    df["_detector_label_raw"] = df["detector_label"]

    # Drift detection metrics are not meaningful for the oracle
    oracle_mask = df["detector"] == "oracle"
    dd_cols = [col for col in df.columns if col[0].startswith("dd.")]
    df.loc[oracle_mask, dd_cols] = np.nan

    def detector_label(row: pd.Series) -> str:
        (strategy,) = row["strategy"].values
        (raw,) = row["_detector_label_raw"].values
        key = (str(boundary), str(strategy), str(raw))
        arrow = UP if key in better else DOWN if key in worse else r"$\ \;$"
        label = raw.replace("ADWIN_JOINT", "ADWIN*").replace("oracle", "Oracle")
        return f"{arrow}{label}"

    out = pd.DataFrame(index=df.index)
    out["Strategy"] = df["strategy"]
    out["Detector"] = df.apply(detector_label, axis=1)
    out[f"Acc. {UP}"] = format_column(df[DV] * 100, max_style="textbf", ndigits=1)
    out[r"Acc$_{ttt}$ " + UP] = format_column(
        df["ttt.accuracy"], max_style="textbf", ndigits=1
    )
    out[f"FWT {UP}"] = format_column(
        df["ocl.forward_transfer"] * 100, max_style="textbf", ndigits=1
    )
    out[f"BWT {UP}"] = format_column(
        df["ocl.backward_transfer"] * 100, max_style="textbf", ndigits=1
    )
    out[f"F1 {UP}"] = format_column(df[F1_METRIC] * 100, max_style="textbf", ndigits=1)
    out[f"NMDT {DOWN}"] = format_column(df["nmdt"], min_style="textbf", ndigits=2)
    return out.set_index(["Strategy", "Detector"])


def write_tables(df_ev: pd.DataFrame, prefix: str) -> None:
    better, worse = significance_tests(df_ev)
    df_grouped = df_ev.groupby(
        ["boundary", "strategy", "detector", "detector_label"]
    ).aggregate(AGGREGATIONS)
    for b in sorted(df_ev["boundary"].unique()):
        table = latex_table(df_grouped.loc[b], b, better, worse).to_latex(
            escape=False, sparsify=True, na_rep=" "
        )
        # No cline at the bottom of the table
        table = table.replace(r"\cline{1-8}", "\\midrule")
        table = table.replace("\\midrule\n\\bottomrule", "\\bottomrule")
        table = table.replace("N/A", "")
        path = TABLE_DIR / f"{prefix}_{b}.tex"
        print(f"Saving {path}")
        path.write_text(table)


if __name__ == "__main__":
    df_ev = pd.read_csv(LOGS / "evaluate/data_frame.csv")
    df_ev["detector_label"] = df_ev["detector_label"].replace(ONE_SIDED_LABELS)
    df_ev["detector_label"] = df_ev["detector_label"].replace("ADWIN_JOINT", "ADWIN*")
    df_ev = compute_nmdt(df_ev)

    # Appendix: all detectors
    write_tables(df_ev, "appendix_strategy")
    # Paper: the control and the two ADWIN variants
    keep = df_ev["detector_label"].isin(["oracle", "ADWIN", "ADWIN*"])
    write_tables(df_ev[keep], "strategy")
