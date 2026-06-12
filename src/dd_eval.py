from dataclasses import asdict
from typing import Any
from capymoa.drift.eval_detector import EvaluateDriftDetector
from src.drift_detector import DriftDetectorArgs, ErrorStreamType, drift_wd
import numpy as np


def evaluate_dd_stream(
    drift_detector_config: DriftDetectorArgs, dd_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one saved error stream with the configured drift detector."""
    evaluator = EvaluateDriftDetector(
        max_delay=dd_metrics["max_delay"],
        max_early_detection=dd_metrics["max_early_detection"],
        rate_period=dd_metrics["rate_period"],
    )
    dd = drift_detector_config.build_dd()

    error_stream_type = ErrorStreamType(drift_detector_config.error_stream_type)
    if error_stream_type == ErrorStreamType.CE:
        stream = dd_metrics["ce_stream"]
    elif error_stream_type == ErrorStreamType.ERROR:
        stream = dd_metrics["error_stream"]
    else:
        raise ValueError(f"Unsupported error stream type: {error_stream_type}")

    preds = []
    for i, element in enumerate(stream):
        dd.add_element(element)
        if dd.detected_change():
            preds.append(i)
            if drift_detector_config.reset_on_drift:
                dd.reset()

    metrics = asdict(
        evaluator.calc_performance(
            dd_metrics["trues"], preds, dd_metrics["tot_n_instances"]
        )
    )
    trues_rel = np.array(dd_metrics["trues"]) / dd_metrics["tot_n_instances"]
    preds_rel = np.array(preds) / dd_metrics["tot_n_instances"]
    metrics["wasserstein_distance"] = drift_wd(trues_rel, preds_rel)
    metrics["trues"] = dd_metrics["trues"]
    metrics["preds"] = preds
    metrics["tot_n_instances"] = dd_metrics["tot_n_instances"]
    return metrics
