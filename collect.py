"""Script to collect result files into a single CSV for analysis."""

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
from src.metrics import drift_confusion
from src.config import converter, Config
import pickle
import yaml
from dataclasses import asdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

# See: https://capymoa.org/api/modules/capymoa.ocl.evaluation.OCLMetrics.html
ocl_metric_keys = [
    "accuracy_final",
    "accuracy_seen_avg",
    "forward_transfer",
    "backward_transfer",
]

# See: https://capymoa.org/api/modules/capymoa.drift.eval_detector.DriftDetectionMetrics.html
dd_metric_keys = [
    "precision",
    "recall",
    "f1",
    "mdt",
    "far",
    "my_f1",
    "my_fp",
    "my_tp",
    "my_fn",
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


def load_dd_run_record(dirname: Path | str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load drift-detector replay results from one dd_run directory."""
    dirname = Path(dirname)
    with open(dirname / "dd_metrics.pkl", "rb") as f:
        dd_metrics = pickle.load(f)

    # Expected layout: logs/??/<boundary>/<strategy>/<detector>/<trial>
    detector = dirname.parent.name
    strategy = dirname.parent.parent.name
    boundary = dirname.parent.parent.parent.name
    seed = int(dirname.name)

    config_filename = dirname / "config.yaml"
    if config_filename.exists():
        with open(config_filename) as f:
            config = converter.structure(yaml.safe_load(f), Config)
            seed = config.seed

    record: Dict[str, int | str | float | bool] = {
        "strategy": strategy,
        "detector": detector,
        "detector_label": detector,
        "boundary": boundary,
        "seed": seed,
    }
    return copy_keys(record, dd_metrics, "dd", dd_metric_keys), dd_metrics


def collect_dataset(dirs: Sequence[Path | str], output_file: Path | str) -> None:
    """Collect the results from multiple directories and save to a CSV file."""
    metadata = []
    error_stream = []
    ce_stream = []
    trues = []
    preds = []
    for dirname in tqdm.tqdm(dirs):
        record, dd_metrics = load_dd_run_record(dirname)
        error_stream.append(dd_metrics["error_stream"])
        ce_stream.append(dd_metrics["ce_stream"])
        trues.append(dd_metrics["trues"])
        preds.append(dd_metrics["preds"])
        metadata.append(record)

    df = pd.DataFrame(metadata)

    seed_array = pa.array(df["seed"], pa.int32())
    strategy_array = pa.array(df["strategy"], pa.string())
    detector_array = pa.array(df["detector"], pa.string())
    boundary_array = pa.array(df["boundary"], pa.string())
    ce_stream_array = pa.array(ce_stream, pa.large_list(pa.float16()))
    error_stream_array = pa.array(error_stream, pa.large_list(pa.bool_()))

    trues_array = pa.array(trues, pa.list_(pa.int32(), 4))
    preds_array = pa.array(preds, pa.list_(pa.int32()))

    table = pa.Table.from_arrays(
        [
            boundary_array,
            strategy_array,
            detector_array,
            seed_array,
            ce_stream_array,
            error_stream_array,
            trues_array,
            preds_array,
        ],
        names=[
            "boundary",
            "strategy",
            "detector",
            "seed",
            "ce_stream",
            "error_stream",
            "trues",
            "preds",
        ],
    )

    # Count number of each strategy
    strategy_counts = df["strategy"].value_counts()
    print(strategy_counts)

    # Save to a parquet file
    pq.write_table(table, output_file)


def load_records(dirs: Sequence[Path | str]) -> pd.DataFrame:
    """Load the results from multiple directories."""
    return pd.DataFrame([load_record(dirname) for dirname in dirs])


def load_dd_run_records(dirs: Sequence[Path | str]) -> pd.DataFrame:
    """Load drift-detector replay results from multiple dd_run directories."""
    return pd.DataFrame([load_dd_run_record(dirname)[0] for dirname in dirs])
