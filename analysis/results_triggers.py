"""Write the trigger table (LaTeX): what each blind trigger or fault keeps of the Oracle's gain.

Run with `uv run analysis/results_triggers.py`. Reads the RotatedFashionMNIST
`evaluate/data_frame.csv` and the `synthetic` runs under `$LOGDIR/cldd/rfmnist`. Writes
`table/triggers.tex`.

Every column is computed on the 10 evaluation seeds, for each of the 15 pairs of strategy
and drift speed. Accuracy is the average accuracy on the seen distributions, in %. The
extra seeds of the `ADWIN_more` and `oracle_more` runs are not used, so that every row
has the same seeds.

* Delta Oracle: mean over the pairs of the paired accuracy difference to the Oracle run
  of the same seed, in points.
* Delta ADWIN: the same, to the ADWIN run of the same seed.
* Kept: the mean gain of the trigger over FT divided by the mean gain of the Oracle over
  FT. Both means are over the pairs, of the accuracies averaged over seeds. It is how
  much of the continual learner's improvement the trigger keeps.

The Oracle and ADWIN are references and have no row. On these seeds ADWIN is 0.04 points
below the Oracle. The Oracle is the one of the evaluate runs, which fires when a task
begins. The `oracle_off0` runs replay the same alarms with the schedule detector, as the
rows here do. They are about 0.6 points below it and are not shown.
"""

import pickle

import numpy as np
import pandas as pd
import yaml
from results_strategy import LOGS, TABLE_DIR

STRATEGIES = ["EWC", "LWF", "MAS", "RWalk", "SI"]
BOUNDARIES = ["abrupt", "gradual", "slow"]
ACC = "ocl.accuracy_seen_avg"
SYNTHETIC = LOGS / "synthetic"

# Rows: (label, synthetic variant) in groups. The streams have about 300,000 instances,
# so the periodic triggers every 15k, 30k, 60k and 120k fire 20, 10, 5 and 2 or 3 times
# (the phase of the first one is random).
GROUPS = [
    [
        ("Triggers $20$ times at equal intervals", "periodic15k"),
        ("Triggers $10$ times at equal intervals", "periodic30k"),
        ("Triggers $5$ times at equal intervals", "periodic60k"),
        ("Triggers $2$--$3$ times at equal intervals", "periodic120k"),
        ("Random, ADWIN's trigger count", "random_adwin"),
    ],
    [
        ("Silent, strategy's learning rate", "oracle_miss4"),
        ("Silent, FT's learning rate", "silent_ftlr"),
    ],
]


def load_synthetic(variant: str) -> dict[tuple[str, str], pd.Series]:
    """Accuracy by seed of one variant, for each (boundary, strategy) cell."""
    out = {}
    for boundary in BOUNDARIES:
        for strategy in STRATEGIES:
            acc = {}
            for run in sorted((SYNTHETIC / boundary / strategy / variant).glob("*")):
                metrics = run / "ocl_metrics.pkl"
                if not metrics.exists():
                    continue
                with open(run / "config.yaml") as f:
                    seed = yaml.safe_load(f)["seed"]
                with open(metrics, "rb") as f:
                    acc[seed] = float(pickle.load(f).accuracy_seen_avg)
            out[(boundary, strategy)] = pd.Series(acc, dtype=float)
    return out


def from_evaluate(df: pd.DataFrame, label: str):
    """Accuracy by seed of an evaluate detector, for each (boundary, strategy) cell."""
    out = {}
    for boundary in BOUNDARIES:
        for strategy in STRATEGIES:
            d = df[
                (df["detector_label"] == label)
                & (df["boundary"] == boundary)
                & (df["strategy"] == strategy)
            ]
            out[(boundary, strategy)] = d.set_index("seed")[ACC]
    return out


def common_seeds(runs: dict[tuple[str, str], pd.Series]) -> list[int]:
    sets = [set(s.index) for s in runs.values()]
    return sorted(set.intersection(*sets)) if sets else []


def cell_means(runs, seeds) -> pd.Series:
    return pd.Series({k: s.loc[seeds].mean() for k, s in runs.items()})


def fmt_delta(x: float) -> str:
    x = round(x, 1) + 0.0  # no negative zero
    return f"${x:+.1f}$".replace("+-", "-") if x else "$0.0$"


def fmt_kept(x: float) -> str:
    pct = round(100 * x)
    return rf"$-{abs(pct)}$\%" if pct < 0 else rf"{pct}\%"


def paired_delta(runs, reference, seeds: list[int]) -> float:
    """Mean over the pairs of the paired accuracy difference to `reference`, in points."""
    return 100 * np.mean(
        [(runs[k].loc[seeds] - reference[k].loc[seeds]).mean() for k in runs]
    )


def summarise(runs, oracle, adwin, ft) -> dict:
    seeds = common_seeds(runs)
    ft_seeds = [s for s in seeds if s in set(ft[BOUNDARIES[0]].index)]
    trigger_gain, oracle_gain = [], []
    for k in runs:
        ft_acc = ft[k[0]].loc[ft_seeds].mean()
        trigger_gain.append(runs[k].loc[ft_seeds].mean() - ft_acc)
        oracle_gain.append(oracle[k].loc[ft_seeds].mean() - ft_acc)
    return {
        "delta_oracle": paired_delta(runs, oracle, seeds),
        "delta_adwin": paired_delta(runs, adwin, seeds),
        "kept": float(np.mean(trigger_gain) / np.mean(oracle_gain)),
    }


def build() -> str:
    df = pd.read_csv(LOGS / "evaluate/data_frame.csv")
    oracle = from_evaluate(df, "oracle")
    adwin = from_evaluate(df, "ADWIN")
    ft_df = df[df["strategy"] == "FT"]
    ft = {b: ft_df[ft_df["boundary"] == b].set_index("seed")[ACC] for b in BOUNDARIES}

    rows = [[(label, load_synthetic(v)) for label, v in group] for group in GROUPS]

    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule"]
    lines.append(r"Trigger & $\Delta$ Oracle & $\Delta$ ADWIN & Kept \\")
    for group in rows:
        lines.append(r"\midrule")
        for label, runs in group:
            s = summarise(runs, oracle, adwin, ft)
            lines.append(
                f"{label} & {fmt_delta(s['delta_oracle'])} & "
                f"{fmt_delta(s['delta_adwin'])} & {fmt_kept(s['kept'])} \\\\"
            )
            print(label, s, len(common_seeds(runs)), "seeds")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    path = TABLE_DIR / "triggers.tex"
    print(f"Saving {path}")
    path.write_text(build())
