"""Paired significance tests for the main claims of the paper.

Run with `uv run analysis/results_significance.py`. Reads the evaluate runs of RFMNIST and
ECovtype and the `synthetic` runs of RFMNIST under `$LOGDIR/cldd`. Prints the results;
nothing is written.

The same seed gives the same initialisation and instance order under every trigger, so runs
are paired by seed. Every test is a two-sided paired t-test over seeds on the average
accuracy on the seen distributions, in points.

* Per-pair tests: one test for each of the 15 pairs (strategy and drift speed). Holm's
  method corrects the 15 p-values of a family, so a family has a 5% chance of any false
  positive.
* Pooled tests: the difference of each seed is first averaged over the 15 pairs, then
  tested over the seeds. This tests a claim about the mean over pairs.

On RFMNIST, ADWIN and the Oracle also have 10 extra seeds (`ADWIN_more`, `oracle_more`);
their comparison uses all 20.
"""

import pandas as pd
from scipy import stats

from common import ALPHA, ONE_SIDED_LABELS, holm
from results_strategy import LOGS
from results_triggers import STRATEGIES, load_synthetic

ACC = "ocl.accuracy_seen_avg"
DETECTORS = ["ADWIN", "ADWIN*", "DDM", "PH", "SEED", "STEPD"]
DATASETS = {"RFMNIST": LOGS, "ECovtype": LOGS.parent / "covertype"}
KEY = ["boundary", "strategy", "seed"]


def evaluate_runs(root) -> pd.DataFrame:
    """Accuracy (points) of every evaluate run, with plain detector labels."""
    df = pd.read_csv(root / "evaluate/data_frame.csv")
    df["detector"] = (
        df["detector_label"]
        .replace(ONE_SIDED_LABELS)
        .replace({"ADWIN_JOINT": "ADWIN*"})
    )
    df["acc"] = 100 * df[ACC]
    return df[[*KEY, "detector", "acc"]]


def runs_of(df: pd.DataFrame, detector: str) -> pd.DataFrame:
    d = df[(df["detector"] == detector) & df["strategy"].isin(STRATEGIES)]
    return d[KEY + ["acc"]]


def synthetic_runs(variant: str) -> pd.DataFrame:
    """Accuracy (points) of a `synthetic` variant, as rows of (boundary, strategy, seed)."""
    rows = [
        {"boundary": b, "strategy": s, "seed": seed, "acc": 100 * acc}
        for (b, s), series in load_synthetic(variant).items()
        for seed, acc in series.items()
    ]
    return pd.DataFrame(rows)


def ft_as_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """FT runs once per drift speed; repeat it for every strategy to pair it by seed."""
    ft = df[(df["strategy"] == "FT") & (df["detector"] == "oracle")]
    return pd.concat(
        [ft[["boundary", "seed", "acc"]].assign(strategy=s) for s in STRATEGIES]
    )


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Difference a - b (points) for every (boundary, strategy, seed) both have."""
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    return m.assign(diff=m["acc_a"] - m["acc_b"])[KEY + ["diff"]]


def per_pair(diff: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test for each pair, with Holm-corrected p-values over the family."""
    rows = []
    for (b, s), g in diff.groupby(["boundary", "strategy"]):
        t = stats.ttest_1samp(g["diff"], 0.0)
        rows.append(
            {
                "boundary": b,
                "strategy": s,
                "n": len(g),
                "mean": g["diff"].mean(),
                "p": t.pvalue,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out.sort_values("p").reset_index(drop=True)


def pooled(diff: pd.DataFrame) -> dict:
    """Paired t-test over seeds of the difference averaged over the pairs."""
    per_seed = diff.groupby("seed")["diff"].mean()
    t = stats.ttest_1samp(per_seed, 0.0)
    lo, hi = t.confidence_interval(1 - ALPHA)
    return {"n": len(per_seed), "mean": per_seed.mean(), "ci": (lo, hi), "p": t.pvalue}


def summary(name: str, diff: pd.DataFrame) -> None:
    pp = per_pair(diff)
    sig = pp[pp["p_holm"] < ALPHA]
    po = pooled(diff)
    print(
        f"{name:42s} pooled {po['mean']:+5.1f} [{po['ci'][0]:+5.1f}, {po['ci'][1]:+5.1f}]"
        f" p={po['p']:.3g} (n={po['n']}) | pairs below: {(sig['mean'] < 0).sum():2d},"
        f" above: {(sig['mean'] > 0).sum():2d} of {len(pp)}"
    )


def main() -> None:
    evaluate = {name: evaluate_runs(root) for name, root in DATASETS.items()}

    print("== Detectors against the Oracle (mid)")
    for name, df in evaluate.items():
        oracle = runs_of(df, "oracle")
        for det in DETECTORS:
            summary(f"{name} {det} - Oracle", paired(runs_of(df, det), oracle))
    # ADWIN on RFMNIST with the 10 extra seeds
    df = evaluate["RFMNIST"]
    adwin20 = pd.concat([runs_of(df, "ADWIN"), synthetic_runs("ADWIN_more")])
    oracle20 = pd.concat([runs_of(df, "oracle"), synthetic_runs("oracle_more")])
    summary("RFMNIST ADWIN - Oracle (20 seeds)", paired(adwin20, oracle20))

    print("== Oracle variants against the mid Oracle")
    for name, df in evaluate.items():
        oracle = runs_of(df, "oracle")
        for variant in ["early", "late"]:
            d = paired(runs_of(df, variant), oracle)
            summary(f"{name} {variant} - mid", d[d["boundary"] != "abrupt"])

    print("== Joint tuning")
    for name, df in evaluate.items():
        summary(
            f"{name} ADWIN* - ADWIN",
            paired(runs_of(df, "ADWIN*"), runs_of(df, "ADWIN")),
        )

    print("== Blind triggers against ADWIN (RFMNIST)")
    adwin = runs_of(evaluate["RFMNIST"], "ADWIN")
    for variant in [
        "periodic15k",
        "periodic30k",
        "periodic60k",
        "periodic120k",
        "random_adwin",
    ]:
        summary(f"{variant} - ADWIN", paired(synthetic_runs(variant), adwin))

    print("== Faults against the Oracle (RFMNIST)")
    oracle = runs_of(evaluate["RFMNIST"], "oracle")
    faults = {
        v: synthetic_runs(v)
        for v in [
            "oracle_off0",
            "oracle_miss1",
            "oracle_miss2",
            "oracle_miss3",
            "oracle_miss4",
            "oracle_extra3",
            "oracle_extra10",
            "oracle_extra20",
            "oracle_off-6k",
            "oracle_off+6k",
        ]
    }
    for v, runs in faults.items():
        summary(f"{v} - Oracle", paired(runs, oracle))
    print("== One miss against random false triggers (RFMNIST)")
    for v in ["oracle_extra3", "oracle_extra10", "oracle_extra20"]:
        summary(f"oracle_miss1 - {v}", paired(faults["oracle_miss1"], faults[v]))

    print("== Silent strategies against FT (RFMNIST)")
    ft = ft_as_pairs(evaluate["RFMNIST"])
    summary("oracle_miss4 - FT", paired(faults["oracle_miss4"], ft))
    summary("silent_ftlr - FT", paired(synthetic_runs("silent_ftlr"), ft))


if __name__ == "__main__":
    main()
