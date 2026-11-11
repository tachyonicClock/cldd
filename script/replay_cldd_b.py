"""Replay each tuned detector on the CLDD-B streams of the other detectors.

Usage (from the repo root):
    uv run python -P script/replay_cldd_b.py --dataset rfmnist --out <csv> [--workers 8]

`-P` keeps `script/` off the import path, where `script/profile.py` would hide the standard
`profile` module.

For every strategy, drift speed and detector D, D runs with its phase-2 hyperparameters
(the same configuration as the `dd_run` task) on the evaluation error streams recorded
while each of the four other detectors controlled the learner, never its own. This is the
open-loop score on detector-controlled streams. It uses `evaluate_dd_stream`, so the
scores are computed as in `dd_run`. One row per replayed stream.
"""

import pickle
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dodo import BOUNDARY, DETECTOR, DETECTOR_AGNOSTIC, EVALUATION_SEEDS, STRATEGY, Unit

METRICS = ["my_f1", "my_tp", "my_fp", "my_fn"]


def replay(job: tuple) -> list[dict]:
    """Replay one detector configuration on the streams of the other detectors."""
    from src.config import get_config
    from src.dd_eval import evaluate_dd_stream

    dataset, strategy, boundary, detector = job
    unit = Unit(strategy, detector, boundary, dataset=dataset)
    config = get_config(unit.configs() + [unit.tune_detector_hp], [])
    rows = []
    for recorder, (trial, seed) in product(DETECTOR, enumerate(EVALUATION_SEEDS)):
        if recorder == detector:
            continue
        stream = Unit(strategy, recorder, boundary, trial, dataset).evaluate_dd_metrics
        with open(stream, "rb") as f:
            dd_metrics = pickle.load(f)
        metrics = evaluate_dd_stream(config.drift_detector, dd_metrics)
        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "boundary": boundary,
                "detector_label": detector,
                "recorder": recorder,
                "seed": seed,
                **{f"dd.{m}": metrics[m] for m in METRICS},
            }
        )
    return rows


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = [
        (args.dataset, s, b, d)
        for s, b, d in product(STRATEGY, BOUNDARY, DETECTOR)
        if s not in DETECTOR_AGNOSTIC
    ]
    # Each worker starts its own JVM for the detectors, so fork is not safe
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = [row for result in pool.map(replay, jobs) for row in result]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.4f")
    print(f"Saved {len(rows)} replays to {args.out}")


if __name__ == "__main__":
    main()
