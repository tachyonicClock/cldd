"""Doit task graph for the drift-detection experiments.

This file implements the three-phase workflow described in ``plan.md``:

1) Tune continual-learning strategy hyper-parameters with an oracle detector.
2) Generate error-rate streams and tune detector hyper-parameters.
3) Evaluate final configurations on held-out random seeds.

The current constants in this file select a reduced subset of strategies,
detectors, boundaries, and seeds suitable for iterative experimentation.
"""

from itertools import product
from pathlib import Path
from dataclasses import dataclass
from actions import (
    tune_strategy,
    error_stream,
    tune_detector,
    select_best_detector,
    evaluate,
    collect_evaluate_records,
)

Strategy = str
Detector = str
Boundary = str

STRATEGY = [
    "EWC",
    "FT",
]
DETECTOR_AGNOSTIC = {"FT"}
DETECTOR = [
    "ADWIN",
    "CUSUM",
    "DDM",
    "SEED",
    "STEPD",
    "PH",
]
BOUNDARY = [
    "abrupt",
    "gradual",
    "slow",
]
ERROR_STREAM_SEEDS = [
    0,
    # 1,
]
EVALUATION_SEEDS = [
    2,
    # 3,
]
ORACLE_DETECTOR = "oracle"
BEST_DETECTOR = "BEST"


@dataclass
class Unit:
    """Represents one experiment unit identified by strategy/detector/boundary/trial.

    The instance also provides helper methods that construct doit task
    dictionaries and canonical file locations used across all phases.
    """

    strategy: Strategy
    detector: Detector
    boundary: Boundary
    trial: int = 0

    @property
    def trial_str(self) -> str:
        return f"{self.trial:03d}"

    @property
    def identifier(self) -> str:
        return f"{self.strategy}.{self.detector}.{self.boundary}.{self.trial_str}"

    @property
    def configs(self) -> list[Path]:
        r = Path("config")
        return [
            r / "base.yml",
            r / "strategy" / f"{self.strategy}.yml",
            r / "detector" / f"{self.detector}.yml",
            r / "boundary" / f"{self.boundary}.yml",
        ]

    def logdir(self, label: str) -> Path:
        return (
            Path("logs")
            / label
            / self.boundary
            / self.strategy
            / self.detector
            / self.trial_str
        )

    @property
    def tune_strategy_hp(self) -> Path:
        return self.logdir(tune_strategy.__name__).parent / "best_params.yml"

    @property
    def tune_detector_hp(self) -> Path:
        return self.logdir(tune_detector.__name__).parent / "best_params.yml"

    @property
    def tune_detector_metrics(self) -> Path:
        return self.logdir(tune_detector.__name__).parent / "best_trial.yml"

    @property
    def error_stream(self) -> Path:
        return self.logdir(error_stream.__name__) / "dd_metrics.pkl"

    @property
    def ocl_metrics(self) -> Path:
        return self.logdir(evaluate.__name__) / "ocl_metrics.pkl"

    def task_tune_strategy(self) -> dict:
        return {
            "name": self.identifier,
            "actions": [(tune_strategy, (self.configs, self.identifier))],
            "file_dep": self.configs,
            "targets": [self.tune_strategy_hp],
        }

    def task_error_stream(self, seed: int):
        configs = self.configs + [self.tune_strategy_hp]
        return {
            "name": self.identifier,
            "actions": [(error_stream, (configs, seed, self.trial, self.identifier))],
            "file_dep": configs,
            "targets": [self.error_stream],
        }

    def task_tune_hp_detector(self, error_streams: list[Path]):
        configs = self.configs
        return {
            "name": self.identifier,
            "actions": [(tune_detector, (configs, error_streams, self.identifier))],
            "file_dep": configs + error_streams,
            "targets": [self.tune_detector_hp, self.tune_detector_metrics],
        }

    def task_select_best_detector(self, trial_files: list[Path]):
        configs = self.configs
        return {
            "name": self.identifier,
            "actions": [(select_best_detector, (trial_files, self.tune_detector_hp))],
            "file_dep": configs + trial_files,
            "targets": [self.tune_detector_hp],
        }

    def task_evaluate(self, seed: int, configs: list[Path]):
        configs = self.configs + configs
        return {
            "name": self.identifier,
            "actions": [
                (
                    evaluate,
                    (configs, seed, self.trial, self.detector, self.identifier),
                )
            ],
            "file_dep": configs,
            "targets": [self.ocl_metrics],
        }


def task_tune_strategy():
    """Phase 1: tune strategy HP using the oracle detector for each boundary."""

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        yield Unit(strategy, ORACLE_DETECTOR, boundary).task_tune_strategy()


def task_error_stream():
    """Phase 1 output: create per-seed error streams using tuned strategy HP."""

    for strategy, boundary, (trial, seed) in product(
        STRATEGY, BOUNDARY, enumerate(ERROR_STREAM_SEEDS)
    ):
        yield Unit(strategy, ORACLE_DETECTOR, boundary, trial).task_error_stream(seed)


def task_tune_hp_detector():
    """Phase 2: tune detector HP from all error streams of a strategy/boundary."""

    for strategy, boundary, detector in product(STRATEGY, BOUNDARY, DETECTOR):
        # Collect error streams for all trials of this strategy and boundary, which will
        # be used for tuning the detector.
        error_streams = []
        for trial, _ in enumerate(ERROR_STREAM_SEEDS):
            error_streams.append(
                Unit(strategy, ORACLE_DETECTOR, boundary, trial).error_stream
            )

        yield Unit(strategy, detector, boundary).task_tune_hp_detector(error_streams)


def task_select_best_detector():
    """Phase 2 selection: choose the best detector from detector trial summaries."""

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        trial_files = []
        for detector in DETECTOR:
            trial_files.append(Unit(strategy, detector, boundary).tune_detector_metrics)
        yield Unit(strategy, BEST_DETECTOR, boundary).task_select_best_detector(
            trial_files
        )


def iter_evaluate_specs():
    """Yield evaluate unit, seed, and extra config dependencies."""

    for strategy, boundary, (trial, seed) in product(
        STRATEGY, BOUNDARY, enumerate(EVALUATION_SEEDS)
    ):
        strategy_hp = Unit(strategy, ORACLE_DETECTOR, boundary).tune_strategy_hp
        detector_hp = Unit(strategy, BEST_DETECTOR, boundary).tune_detector_hp

        yield Unit(strategy, ORACLE_DETECTOR, boundary, trial), seed, [strategy_hp]
        if strategy not in DETECTOR_AGNOSTIC:
            yield (
                Unit(strategy, BEST_DETECTOR, boundary, trial),
                seed,
                [strategy_hp, detector_hp],
            )


def task_evaluate():
    """Phase 3: evaluate tuned configurations on held-out evaluation seeds.

    Detector-agnostic strategies are evaluated once with the oracle boundary
    setting, while boundary-aware strategies are evaluated with the selected
    best detector as well.
    """

    for unit, seed, configs in iter_evaluate_specs():
        yield unit.task_evaluate(seed, configs)


def task_collect_evaluate():
    """Collect all evaluate outputs into one CSV for analysis."""

    evaluate_dirs: list[Path] = []
    file_deps: list[Path] = []

    for unit, _, _ in iter_evaluate_specs():
        evaluate_dirs.append(unit.logdir(evaluate.__name__))
        file_deps.append(unit.ocl_metrics)

    target = Path("logs") / evaluate.__name__ / "data_frame.csv"
    return {
        "actions": [(collect_evaluate_records, (evaluate_dirs, target))],
        "file_dep": file_deps + ["collect.py"],
        "targets": [target],
    }
