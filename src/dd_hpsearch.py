"""Perform hyperparameter search for a drift detector using a frozen error-rate stream
from a previous experiment. This allows us to optimize drift detector hyperparameters
without the computational cost of running a full experiment for each trial.
"""

from src.config import Config
from src.drift_detector import ErrorStreamType
from src.hpsearch import study_name_from_config, optimize_with_max_trials
from src.util import obj_dot_notation_set
import optuna
import re
import pickle
from pathlib import Path
from loguru import logger
from pprint import pprint
from dataclasses import asdict, fields, is_dataclass


class DDHPSearch:
    def __init__(self, config: Config, source: str):
        assert config.hpsearch is not None
        self.config = config
        self.hpsearch = config.hpsearch

        base_attrs = self._load_source_metadata(source)

        study_name = study_name_from_config(config)
        logger.info(f"Setup study `{study_name}`.")
        self.study = optuna.create_study(
            study_name=study_name,
            storage=config.hpsearch.storage,
            direction="maximize",
            load_if_exists=True,
        )
        for key, value in base_attrs.items():
            self.study.set_user_attr(key, value)

    def _load_source_metadata(self, source: str) -> dict[str, str | dict]:
        source_path = Path(source)
        if source_path.exists():
            self.dd_metrics_runs = self._load_dd_metrics(source_path)
            return {
                "dd_metrics_source": source_path.as_posix(),
                "base_study.best_trial.params": self._config_params(),
            }

        base_study = optuna.load_study(study_name=source, storage=self.hpsearch.storage)
        trial_number = base_study.best_trial.number

        match = re.match(r"bocl/([^/]+)/([^/]+)/([^/]+)", source)
        if not match:
            raise ValueError(f"Study name `{source}` does not match expected format.")

        label, scenario, method = match.groups()
        dd_metrics_path = Path(
            f"logs/{label}/{scenario}/{method}/{trial_number:03d}/dd_metrics.pkl"
        )
        self.dd_metrics_runs = self._load_dd_metrics(dd_metrics_path)
        return {
            "base_study": source,
            "base_study.best_trial.params": base_study.best_trial.params,
            "dd_metrics_source": dd_metrics_path.as_posix(),
        }

    def _config_params(self) -> dict[str, object]:
        params = {}
        for section_name in ("learner", "model"):
            section = getattr(self.config, section_name)
            if not is_dataclass(section):
                continue
            for field in fields(section):
                params[f"{section_name}.{field.name}"] = getattr(section, field.name)
        return params

    def _load_dd_metrics(self, path: Path) -> list[dict]:
        with path.open("rb") as f:
            dd_metrics = pickle.load(f)

        if isinstance(dd_metrics, list):
            return dd_metrics
        if isinstance(dd_metrics, dict):
            return [dd_metrics]

        raise TypeError(
            f"Unsupported dd_metrics payload in `{path}`: {type(dd_metrics)}"
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

    def optimize(self) -> None:
        optimize_with_max_trials(
            self.study,
            self._objective,
            n_trials=self.hpsearch.n_trials,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        assert not self.config.drift_detector.use_batch_mean, (
            "Batch mean is not supported for DDHPSearch."
        )

        # Apply suggestions to config.
        # ONLY OPTIMIZE DRIFT DETECTOR HYPERPARAMETERS
        suggestions = self.hpsearch.suggest(trial, "drift_detector")
        for key, value in suggestions.items():
            obj_dot_notation_set(key, self.config, value)
        logger.info(f"Trial {trial.number} with suggestions:")
        logger.info("{}", suggestions)

        self.config.trial = trial.number
        self.config.seed = trial.number

        metrics_per_stream = [
            self._evaluate_stream(dd_metrics) for dd_metrics in self.dd_metrics_runs
        ]
        summary_metrics = self._mean_metrics(metrics_per_stream)
        logger.info("{}", summary_metrics)
        trial.set_user_attr("dd_metrics", summary_metrics)
        trial.set_user_attr("dd_metrics_per_stream", metrics_per_stream)

        return -summary_metrics["wasserstein_distance"]
