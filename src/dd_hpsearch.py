"""Perform hyperparameter search for a drift detector using a frozen error-rate stream
from a previous experiment. This allows us to optimize drift detector hyperparameters
without the computational cost of running a full experiment for each trial.
"""

from src.config import Config
from src.drift_detector import ErrorStreamType
from src.hpsearch import optimize_with_max_trials, recreate_study
from src.util import obj_dot_notation_set
import optuna
import pickle
from pathlib import Path
from loguru import logger
from dataclasses import asdict


class DDHPSearch:
    def __init__(self, config: Config, error_streams: list[Path]):
        assert config.hpsearch is not None
        self.config = config
        self.hpsearch = config.hpsearch
        self.error_streams = error_streams
        self.study = recreate_study(
            study_name=config.study_name,
            storage=config.hpsearch.storage,
        )

    def _evaluate_stream(self, dd_metrics: dict) -> dict:
        evaluator = self.config.drift_detector.build_dd_evaluator()
        dd = self.config.drift_detector.build_dd()

        error_stream_type = ErrorStreamType(
            self.config.drift_detector.error_stream_type
        )
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
                if self.config.drift_detector.reset_on_drift:
                    dd.reset()

        metrics = asdict(
            evaluator.calc_performance(
                dd_metrics["trues"], preds, dd_metrics["tot_n_instances"]
            )
        )
        metrics["trues"] = dd_metrics["trues"]
        metrics["preds"] = preds
        metrics["tot_n_instances"] = dd_metrics["tot_n_instances"]
        return metrics

    def _mean_metrics(self, metrics_per_stream: list[dict]) -> dict[str, float]:
        keys = metrics_per_stream[0].keys()
        summary = {}
        for key in keys:
            if key in {"trues", "preds"}:
                continue
            summary[key] = float(
                sum(float(metrics[key]) for metrics in metrics_per_stream)
                / len(metrics_per_stream)
            )
        summary["n_streams"] = len(metrics_per_stream)
        return summary

    def optimize(self) -> dict:
        optimize_with_max_trials(
            self.study,
            self._objective,
            n_trials=self.hpsearch.n_trials,
        )
        return self.study.best_trial.params

    def _objective(self, trial: optuna.Trial) -> float:
        # Apply suggestions to config.
        # ONLY OPTIMIZE DRIFT DETECTOR HYPERPARAMETERS
        suggestions = self.hpsearch.suggest(trial, "drift_detector")
        for key, value in suggestions.items():
            obj_dot_notation_set(key, self.config, value)
        logger.info(f"Trial {trial.number} with suggestions:")
        logger.info("{}", suggestions)

        self.config.trial = trial.number
        self.config.seed = trial.number

        metrics_per_stream = []
        for error_stream_file in self.error_streams:
            with open(error_stream_file, "rb") as f:
                dd_metrics = pickle.load(f)
            metrics = self._evaluate_stream(dd_metrics)
            metrics_per_stream.append(metrics)

        summary_metrics = self._mean_metrics(metrics_per_stream)
        trial.set_user_attr("dd_metrics", summary_metrics)
        return -summary_metrics["wasserstein_distance"]
