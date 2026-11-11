"""Does the open-loop F1 rank detectors as the closed loop does?

Run with `uv run analysis/kendal_tau.py` from the repository root. Prints the results;
nothing is written.

For each dataset, drift speed and strategy (a pair), the five detectors are ranked by
their mean over the evaluation seeds of:

* open-loop F1 on the CLDD-A streams of the evaluation seeds (`dd_run`),
* open-loop F1 on the CLDD-B streams of the four other detectors (`replay_cldd_b`),
* closed-loop F1 and closed-loop Acc (`evaluate`).

Each open-loop ranking is compared with each closed-loop ranking by Kendall's tau-b (ties
allowed) and by top-1 agreement (the best open-loop detector is also the best closed-loop
one). A pair whose ranking is all ties has no tau and is left out of the mean. ADWIN* has
no open-loop streams, so it is left out.

Chance: with five detectors top-1 agreement is 1/5 per pair (3 of 15) and the mean tau
is 0. The p-value is a one-sided permutation test: the open-loop scores are shuffled
among the detectors of each pair, independently, and the mean tau recomputed.
"""

import os
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import ONE_SIDED_LABELS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dodo import EVALUATION_SEEDS

DETECTORS = ["ADWIN", "DDM", "PH", "SEED", "STEPD"]
PAIR = ["boundary", "strategy"]
N_PERM = 10_000
SEED = 0


def detector_means(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean of `column` per pair (rows) and detector (columns)."""
    df = df.assign(detector_label=df["detector_label"].replace(ONE_SIDED_LABELS))
    df = df[df["detector_label"].isin(DETECTORS)]
    return df.groupby([*PAIR, "detector_label"])[column].mean().unstack()[DETECTORS]


def load(root: Path) -> dict[str, pd.DataFrame]:
    ev = pd.read_csv(root / "evaluate/data_frame.csv")
    ev = ev[ev["seed"].isin(EVALUATION_SEEDS)]
    oracle = pd.read_csv(root / "dd_run/data_frame.csv")
    cldd_b = pd.read_csv(root / "replay_cldd_b/data_frame.csv")
    return {
        "open F1, CLDD-A": detector_means(oracle, "dd.my_f1"),
        "open F1, CLDD-B": detector_means(cldd_b, "dd.my_f1"),
        "closed F1": detector_means(ev, "dd.my_f1"),
        "closed Acc": detector_means(ev, "ocl.accuracy_seen_avg"),
    }


def taus(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Kendall's tau-b of each pair (row); NaN when a ranking is all ties."""
    return np.array(
        [stats.kendalltau(a.loc[i], b.loc[i]).statistic for i in a.index], dtype=float
    )


def top1(a: pd.DataFrame, b: pd.DataFrame) -> int:
    return int((a.idxmax(axis=1) == b.idxmax(axis=1)).sum())


def perm_p(a: pd.DataFrame, b: pd.DataFrame, observed: float, rng) -> float:
    """One-sided p-value of the mean tau, shuffling `a` within each pair.

    Five detectors have only 120 orders, so the tau of every order is computed once per
    pair and the random shuffles index into that table.
    """
    orders = list(permutations(range(len(DETECTORS))))
    table = np.array(
        [
            [stats.kendalltau(x[list(o)], y).statistic for o in orders]
            for x, y in zip(a.to_numpy(), b.to_numpy())
        ],
        dtype=float,
    )
    draws = rng.integers(len(orders), size=(N_PERM, len(table)))
    means = np.nanmean(table[np.arange(len(table)), draws], axis=1)
    return (np.sum(means >= observed) + 1) / (N_PERM + 1)


def main() -> None:
    logs_path = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
    rng = np.random.default_rng(SEED)
    for name, dataset in [("RFMNIST", "rfmnist"), ("ECovtype", "covertype")]:
        ranks = load(logs_path / dataset)
        index = ranks["closed Acc"].index
        ranks = {k: v.reindex(index) for k, v in ranks.items()}
        print(f"== {name} ({len(index)} pairs)")
        for open_ in ["open F1, CLDD-A", "open F1, CLDD-B"]:
            for closed in ["closed Acc", "closed F1"]:
                t = taus(ranks[open_], ranks[closed])
                mean = np.nanmean(t)
                p = perm_p(ranks[open_], ranks[closed], mean, rng)
                print(
                    f"  {open_} vs {closed:10s}: top-1 {top1(ranks[open_], ranks[closed]):2d}"
                    f" of {len(index)}, mean tau {mean:+.2f} (median {np.nanmedian(t):+.2f},"
                    f" {np.sum(~np.isnan(t))} pairs), p={p:.4f}"
                )
        # Per drift speed, for the comparison the paper reports
        for open_ in ["open F1, CLDD-A", "open F1, CLDD-B"]:
            t = pd.Series(taus(ranks[open_], ranks["closed Acc"]), index=index)
            by_speed = t.groupby(level="boundary").mean()
            print(
                f"  {open_} vs closed Acc, mean tau by speed: "
                + ", ".join(f"{b} {v:+.2f}" for b, v in by_speed.items())
            )


if __name__ == "__main__":
    main()
