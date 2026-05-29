"""Script to collect result files into a single CSV for analysis."""

from pathlib import Path
from typing import Dict, Sequence, Tuple
from src.config import converter, Config
import pickle
import yaml
from dataclasses import asdict
import pandas as pd

# See: https://capymoa.org/api/modules/capymoa.ocl.evaluation.OCLMetrics.html
ocl_metric_keys = [
    "accuracy_all_avg",
    "accuracy_seen_avg",
    "forward_transfer",
    "backward_transfer",
]

# See: https://capymoa.org/api/modules/capymoa.drift.eval_detector.DriftDetectionMetrics.html
dd_metric_keys = [
    "fp",
    "tp",
    "fn",
    "precision",
    "recall",
    "f1",
    "wasserstein_distance",
    "mdt",
    "far",
]

ttt_metric_keys = ["accuracy"]


def copy_keys(
    dst: dict, src: dict, prefix: str, keys: Sequence[Tuple[str, str] | str]
) -> dict:
    for key in keys:
        if isinstance(key, str):
            dst[f"{prefix}.{key}"] = float(src[key])
        else:
            dst[f"{prefix}.{key[1]}"] = float(src[key[0]])
    return dst


def load_record(dirname: Path | str) -> Dict[str, int | str | float | bool]:
    """Load the results from a directory."""
    dirname = Path(dirname)
    with open(dirname / "config.yaml") as f:
        config = converter.structure(yaml.safe_load(f), Config)
    with open(dirname / "dd_metrics.pkl", "rb") as f:
        dd_metrics = pickle.load(f)
    with open(dirname / "ocl_metrics.pkl", "rb") as f:
        ocl_metrics = asdict(pickle.load(f))
    with open(dirname / "ttt_metrics.pkl", "rb") as f:
        ttt_metrics = pickle.load(f)

    record = {}
    # Index
    record["strategy"] = config.learner.type_
    record["detector"] = config.drift_detector.type_
    record["detector_label"] = config.drift_detector.label
    record["boundary"] = config.scenario.label
    record["seed"] = config.seed

    # Drift Detection Metrics
    record = copy_keys(record, dd_metrics, "dd", dd_metric_keys)
    record = copy_keys(record, ocl_metrics, "ocl", ocl_metric_keys)
    record = copy_keys(record, ttt_metrics["cumulative"], "ttt", ttt_metric_keys)
    return record


def load_records(dirs: Sequence[Path | str]) -> pd.DataFrame:
    """Load the results from multiple directories."""
    return pd.DataFrame([load_record(dirname) for dirname in dirs])
