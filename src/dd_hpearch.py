"""Perform hyperparameter search for a drift detector using a frozen error-rate stream
from a previous experiment. This allows us to optimize drift detector hyperparameters
without the computational cost of running a full experiment for each trial.
"""
from src.config import Config
from src.hpsearch import study_name_from_config, optimize_with_max_trials
from src.util import obj_dot_notation_set
import optuna
import re
import pickle
from loguru import logger
from pprint import pprint


class DDHPSearch:
    def __init__(self, config: Config, study_name: str):
        assert config.hpsearch is not None
        self.config = config
        self.hpsearch = config.hpsearch

        study = optuna.load_study(
            study_name=study_name, storage=config.hpsearch.storage
        )
        trial_number = study.best_trial.number

        # bocl/hp/RotatedTinyMNIST_0.0/EWC_oracle_MLP
        # bocl/{}/{}/{}_{}_{}
        # Extract label, scenario, learner, drift detector, and model from the study name using
        # regex.
        match = re.match(r"bocl/([^/]+)/([^/]+)/([^/]+)", study_name)
        if match:
            label, scenario, method = match.groups()
            learner, drift_detector, model = method.split("_")
        else:
            raise ValueError(
                f"Study name `{study_name}` does not match expected format."
            )

        # logs/hp/RotatedTinyMNIST_0.0/EWC_oracle_MLP/004/dd_metrics.pkl
        with open(
            f"logs/{label}/{scenario}/{method}/{trial_number:03d}/dd_metrics.pkl", "rb"
        ) as f:
            self.dd_metrics = pickle.load(f)

        study_name = study_name_from_config(config)
        logger.info(f"Setup study `{study_name}`.")
        self.study = optuna.create_study(
            study_name=study_name,
            storage=config.hpsearch.storage,
            direction="maximize",
            load_if_exists=True,
        )

    def optimize(self) -> None:
        optimize_with_max_trials(
            self.study,
            self._objective,
            n_trials=self.hpsearch.n_trials,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        # Apply suggestions to config.
        # ONLY OPTIMIZE DRIFT DETECTOR HYPERPARAMETERS
        suggestions = self.hpsearch.suggest(trial, "drift_detector")
        for key, value in suggestions.items():
            obj_dot_notation_set(key, self.config, value)
        logger.info(f"Trial {trial.number} with suggestions:")
        pprint(suggestions)

        self.config.trial = trial.number
        self.config.seed = trial.number

        evaluator = self.config.drift_detector.build_dd_evaluator()
        dd = self.config.drift_detector.build_dd()

        if self.config.drift_detector.error_stream_type == "CE":
            stream = self.dd_metrics["ce_stream"]
        elif self.config.drift_detector.error_stream_type == "ERROR":
            stream = self.dd_metrics["error_stream"]
        else:
            raise ValueError(
                f"Unknown error stream type `{self.config.drift_detector.error_stream_type}`."
            )

        preds = []
        for i, element in enumerate(stream):
            dd.add_element(element)
            if dd.detected_change():
                preds.append(i)

        print(f"TRUES: {self.dd_metrics['trues']}")
        print(f"PREDS: {preds}")

        dd_metrics = evaluator.calc_performance(
            self.dd_metrics["trues"], preds, self.dd_metrics["tot_n_instances"]
        )
        pprint(dd_metrics)
        return dd_metrics.f1
