"""Export UCI Covertype (id 31) to data/covertype_<attr>.pt, one domain per bin of `attr`.

`Covertype` in `src/scenario.py` reads this file. The label is the cover type (7 classes).
The partitioning column(s) are removed from the features; numerics are standardized.
Each domain is subsampled to at most `--per-domain` rows and split 80/20 with a fixed seed.
`--select` ranks candidate attributes by cross-domain concept shift: one logistic
regression per domain, scored by the mean drop in test accuracy on other domains relative
to the in-domain model (higher means more forgetting for FT).

Usage (from the repo root):
    uv run python -P script/covertype.py --select
    uv run python -P script/covertype.py --by Elevation
"""

from os import environ
from pathlib import Path

environ.setdefault("OMP_NUM_THREADS", "1")  # multi-threaded BLAS is slower on small fits
environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import urllib.request
import zipfile
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
NUMERIC = ["Elevation", "Aspect", "Slope", "Horz_Dist_Hydrology", "Vert_Dist_Hydrology",
           "Horz_Dist_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horz_Dist_Fire"]
NAMES = [*NUMERIC, *[f"Wilderness_{i}" for i in range(4)], *[f"Soil_{i}" for i in range(40)], "Cover_Type"]
BINNED = ["Elevation", "Aspect", "Slope", "Horz_Dist_Roadways", "Horz_Dist_Hydrology"]
CANDIDATES = [*BINNED, "Wilderness"]
N_BINS = 5
SEED = 0


def load(root: Path) -> pd.DataFrame:
    archive = root / "covertype.zip"
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        return pd.read_csv(zf.open("covtype.data.gz"), compression="gzip", header=None, names=NAMES)


def domains(df: pd.DataFrame, attr: str) -> tuple[np.ndarray, list]:
    if attr == "Wilderness":
        return df[[f"Wilderness_{i}" for i in range(4)]].to_numpy().argmax(1), [f"area{i}" for i in range(4)]
    edges = np.unique(df[attr].quantile(np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return np.searchsorted(edges, df[attr].to_numpy(), side="right"), [f"bin{i}" for i in range(len(edges) + 1)]


def features(df: pd.DataFrame, attr: str) -> np.ndarray:
    drop = [c for c in df.columns if c.startswith("Wilderness_")] if attr == "Wilderness" else [attr]
    cols = [c for c in df.columns if c not in ("Cover_Type", *drop)]
    x = df[cols].astype(float)
    numeric = [c for c in cols if c in NUMERIC]
    x[numeric] = (x[numeric] - x[numeric].mean()) / (x[numeric].std() + 1e-8)
    return x.to_numpy(np.float32)


def split(domain: np.ndarray, k: int, per_domain: int):
    """Subsample each domain to `per_domain` rows, then split 80/20."""
    rng = np.random.default_rng(SEED)
    tr, te = [], []
    for d in range(k):
        rows = rng.permutation(np.flatnonzero(domain == d))[:per_domain]
        n_te = len(rows) // 5
        te.append(rows[:n_te])
        tr.append(rows[n_te:])
    return tr, te


def shift(df: pd.DataFrame, attr: str, per_domain: int) -> tuple[float, int, list]:
    domain, labels = domains(df, attr)
    x, y = features(df, attr), df["Cover_Type"].to_numpy() - 1
    tr, te = split(domain, len(labels), per_domain)
    k = len(labels)
    acc = np.zeros((k, k))
    for i in range(k):
        model = LogisticRegression(max_iter=200).fit(x[tr[i]], y[tr[i]])
        for j in range(k):
            acc[i, j] = model.score(x[te[j]], y[te[j]])
    gap = np.diag(acc)[None, :] - acc
    np.fill_diagonal(gap, np.nan)
    return float(np.nanmean(gap)), k, labels


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--by", choices=CANDIDATES)
    parser.add_argument("--per-domain", type=int, default=25_000)
    parser.add_argument("--root", type=Path, default=Path(environ["DATASETS"]))
    args = parser.parse_args()
    df = load(args.root)

    if args.select:
        rows = [(a, *shift(df, a, args.per_domain)) for a in CANDIDATES]
        print(f"{'attribute':20} {'gap':>7} {'domains':>7}")
        for a, gap, k, _ in sorted(rows, key=lambda r: -r[1]):
            print(f"{a:20} {gap:7.4f} {k:7d}")
        return
    if args.by is None:
        parser.error("pass --select or --by")

    domain, labels = domains(df, args.by)
    x = torch.from_numpy(features(df, args.by))
    y = torch.from_numpy(df["Cover_Type"].to_numpy() - 1)
    tr, te = split(domain, len(labels), args.per_domain)
    data = {"train": [(x[i], y[i]) for i in tr], "test": [(x[i], y[i]) for i in te], "n_classes": 7}
    for d, label in enumerate(labels):
        print(f"domain {d} ({label}): train {len(tr[d])}, test {len(te[d])}, "
              f"class shares {np.round(np.bincount(y[tr[d]].numpy(), minlength=7) / len(tr[d]), 2).tolist()}")
    out = Path(__file__).resolve().parent.parent / "data" / f"covertype_{args.by}.pt"
    torch.save(data, out)
    print(f"Saved {out} ({x.shape[1]} features)")


if __name__ == "__main__":
    main()
