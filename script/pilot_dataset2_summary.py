"""Summarise the dataset-2 pilots: Acc per strategy and gain over FT (3 seeds, abrupt, Oracle)."""
import os, pickle
from pathlib import Path
import numpy as np, pandas as pd, yaml

LOGS = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
rows = []
for d in LOGS.glob("pilot_*/evaluate/abrupt/*/oracle/*/ocl_metrics.pkl"):
    cand = d.parts[-7].removeprefix("pilot_"); strat = d.parts[-4]
    o = pickle.load(open(d, "rb"))
    b = d.with_name("balanced_metrics.pkl")
    b = pickle.load(open(b, "rb")) if b.exists() else {}
    rows.append(dict(candidate=cand, strategy=strat, acc=100 * o.accuracy_seen_avg,
                     final=100 * o.accuracy_final, bwt=100 * o.backward_transfer,
                     bacc=100 * b.get("balanced_accuracy_seen_avg", np.nan),
                     bbwt=100 * b.get("balanced_backward_transfer", np.nan)))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results yet")
g = df.groupby(["candidate", "strategy"]).agg(n=("acc", "size"), acc=("acc", "mean"), sd=("acc", "std"),
                                              final=("final", "mean"), bwt=("bwt", "mean"),
                                              bacc=("bacc", "mean"), bbwt=("bbwt", "mean"))
ft = g.xs("FT", level="strategy")["acc"]
g["gain_vs_ft"] = g["acc"] - g.index.get_level_values(0).map(ft)
for p in LOGS.glob("pilot_*/tune_strategy/abrupt/*/oracle/best_params.yml"):
    cand = p.parts[-6].removeprefix("pilot_"); strat = p.parts[-3]
    hp = yaml.safe_load(p.read_text())
    if (cand, strat) in g.index:
        g.loc[(cand, strat), "lr"] = hp.get("optimizer", {}).get("lr")
pd.set_option("display.width", 200)
print(g.round(4).to_string())
summ = g.drop(index="FT", level="strategy").groupby(level=0).agg(
    beat_ft=("gain_vs_ft", lambda x: int((x > 0).sum())), beat_ft_by1=("gain_vs_ft", lambda x: int((x > 1).sum())),
    min_gain=("gain_vs_ft", "min"), mean_gain=("gain_vs_ft", "mean"))
print(summ.round(2).to_string())
