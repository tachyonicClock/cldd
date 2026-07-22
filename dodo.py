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
    dd_run,
    evaluate,
    collect_evaluate_records,
    collect_dd_run_records,
)
from collect import collect_dataset
import os
from loguru import logger
import random

DEBUG_MODE = bool(os.environ.get("DEBUG_BOCL", False))
if DEBUG_MODE:
    logger.warning(
        "Running in DEBUG MODE: reduced strategies/detectors/boundaries/seeds."
    )

Strategy = str
Detector = str
Boundary = str

STRATEGY = [
    "EWC",
    "FT",
    "LWF",
    "MAS",
    "RWalk",
    "SI",
]
DETECTOR_AGNOSTIC = {"FT"}
DETECTOR = [
    "ADWIN",
    "DDM",
    "PH",
    "SEED",
    "STEPD",
]
BOUNDARY = [
    "abrupt",
    "gradual",
    "slow",
]

JOINT_HP_DETECTOR = "ADWIN_JOINT"
"""The detector used for joint tuning of strategy and detector HP."""

N_TRIALS = 10
rng = random.Random(0)
rng2 = random.Random(1)
ERROR_STREAM_SEEDS = [rng.randint(0, 10000) for _ in range(N_TRIALS)]
EVALUATION_SEEDS = [rng2.randint(0, 10000) for _ in range(N_TRIALS)]

if DEBUG_MODE:
    ERROR_STREAM_SEEDS = [0]
    EVALUATION_SEEDS = [5]
    STRATEGY = ["FT", "EWC"]
    DETECTOR = ["ADWIN"]
    BOUNDARY = ["abrupt"]

ORACLE_DETECTOR = "oracle"

LOG_ROOT = Path("logs")


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
        _id_path = [self.strategy, self.detector, self.boundary, self.trial_str]
        return ".".join(_id_path)

    def configs(self, hp: str | None = None) -> list[Path]:
        r = Path("config")
        configs = [
            r / "base.yml",
            r / "strategy" / f"{self.strategy}.yml",
            r / "detector" / f"{self.detector}.yml",
            r / "boundary" / f"{self.boundary}.yml",
        ]
        if hp is not None:
            configs.append(r / "hp" / f"{hp}.yml")
        if DEBUG_MODE:
            configs.append(r / "debug.yml")
        return configs

    def logdir(self, label: str) -> Path:
        return (
            LOG_ROOT
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

    @property
    def evaluate_dd_metrics(self) -> Path:
        return self.logdir(evaluate.__name__) / "dd_metrics.pkl"

    @property
    def dd_run_metrics(self) -> Path:
        return self.logdir(dd_run.__name__) / "dd_metrics.pkl"

    def task_tune_strategy(self) -> dict:
        configs = self.configs("strategy")
        return {
            "name": self.identifier,
            "actions": [(tune_strategy, (configs, self.identifier))],
            "file_dep": configs,
            "targets": [self.tune_strategy_hp],
        }

    def task_error_stream(self, seed: int):
        configs = self.configs() + [self.tune_strategy_hp]
        return {
            "name": self.identifier,
            "actions": [(error_stream, (configs, seed, self.trial, self.identifier))],
            "file_dep": configs,
            "targets": [self.error_stream],
        }

    def task_tune_hp_detector(self, error_streams: list[Path]):
        configs = self.configs("detector")
        return {
            "name": self.identifier,
            "actions": [(tune_detector, (configs, error_streams, self.identifier))],
            "file_dep": configs + error_streams,
            "targets": [self.tune_detector_hp, self.tune_detector_metrics],
        }

    def task_evaluate(self, seed: int, configs: list[Path]):
        configs = self.configs() + configs
        return {
            "name": self.identifier,
            "actions": [
                (
                    evaluate,
                    (configs, seed, self.trial, self.detector, self.identifier),
                )
            ],
            "file_dep": configs,
            "targets": [self.ocl_metrics, self.evaluate_dd_metrics],
        }

    def task_dd_run(
        self,
        oracle_error_stream: Path,
        detector_configs: list[Path],
    ):
        configs = self.configs() + detector_configs
        return {
            "name": self.identifier,
            "actions": [
                (
                    dd_run,
                    (
                        configs,
                        oracle_error_stream,
                        self.trial,
                        self.detector,
                        self.identifier,
                    ),
                )
            ],
            "file_dep": configs + [oracle_error_stream],
            "targets": [self.dd_run_metrics],
        }


def task_tune_strategy():
    """Phase 1: tune strategy HP using the oracle detector for each boundary."""

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        yield Unit(strategy, ORACLE_DETECTOR, boundary).task_tune_strategy()


def task_tune_detector_strategy():
    for strategy, boundary in product(STRATEGY, BOUNDARY):
        if strategy in DETECTOR_AGNOSTIC:
            continue
        yield Unit(strategy, JOINT_HP_DETECTOR, boundary).task_tune_strategy()


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


def iter_evaluate_specs():
    """Yield evaluate unit, seed, and extra config dependencies."""

    for strategy, boundary, (trial, seed) in product(
        STRATEGY, BOUNDARY, enumerate(EVALUATION_SEEDS)
    ):
        strategy_hp = Unit(strategy, ORACLE_DETECTOR, boundary).tune_strategy_hp

        yield Unit(strategy, ORACLE_DETECTOR, boundary, trial), seed, [strategy_hp]
        if strategy not in DETECTOR_AGNOSTIC:
            for detector in DETECTOR:
                detector_hp = Unit(strategy, detector, boundary).tune_detector_hp
                yield (
                    Unit(strategy, detector, boundary, trial),
                    seed,
                    [strategy_hp, detector_hp],
                )

            joint_detector_hp = Unit(
                strategy, JOINT_HP_DETECTOR, boundary
            ).tune_strategy_hp
            yield (
                Unit(strategy, JOINT_HP_DETECTOR, boundary, trial),
                seed,
                [joint_detector_hp],
            )


def task_evaluate():
    """Phase 3: evaluate tuned configurations on held-out evaluation seeds.

    Detector-agnostic strategies are evaluated once with the oracle boundary
    setting, while boundary-aware strategies are evaluated with the selected
    best detector as well.
    """

    for unit, seed, configs in iter_evaluate_specs():
        yield unit.task_evaluate(seed, configs)


def task_dd_run():
    """Replay tuned detector configs on oracle evaluate error streams."""

    for strategy, boundary, detector, (trial, _) in product(
        STRATEGY, BOUNDARY, DETECTOR, enumerate(EVALUATION_SEEDS)
    ):
        detector_hp = Unit(strategy, detector, boundary).tune_detector_hp
        oracle_metrics = Unit(
            strategy,
            ORACLE_DETECTOR,
            boundary,
            trial,
        ).evaluate_dd_metrics

        yield Unit(strategy, detector, boundary, trial).task_dd_run(
            oracle_metrics,
            [detector_hp],
        )


def task_collect_evaluate():
    """Collect all evaluate outputs into one CSV for analysis."""

    evaluate_dirs: list[Path] = []
    file_deps: list[Path] = []

    for unit, _, _ in iter_evaluate_specs():
        evaluate_dirs.append(unit.logdir(evaluate.__name__))
        file_deps.append(unit.ocl_metrics)

    target = LOG_ROOT / evaluate.__name__ / "data_frame.csv"
    output_html = LOG_ROOT / "profile" / "evaluate.html"
    return {
        "actions": [
            (collect_evaluate_records, (evaluate_dirs, target)),
            f"uv run script/profile.py 'Evaluation Report' {target} {output_html}",
        ],
        "file_dep": file_deps + ["collect.py", "script/profile.py"],
        "targets": [target, output_html],
    }


def task_CLDD_A():
    """Generate continual learners as a drift detection dataset.

    CLDD-A contains the `oracle` drift detector which perflectly predicts drifts controling
    a continual learning algorithm.
    """
    directories: list[Path] = []
    dependencies: list[Path] = []

    for unit, _, _ in iter_evaluate_specs():
        if unit.detector != ORACLE_DETECTOR:
            continue

        directories.append(unit.logdir(evaluate.__name__))
        dependencies.append(unit.ocl_metrics)

    target = LOG_ROOT / "CLDD_A.parquet"
    return {
        "actions": [
            (collect_dataset, (directories, target)),
        ],
        "file_dep": dependencies + ["collect.py"],
        "targets": [target],
    }


def task_CLDD_B():
    """Generate continual learners as a drift detection dataset.

    CLDD-B contains non-oracle drift detectors that imperfectly controlling a
    continual learning algorithm."""
    directories: list[Path] = []
    dependencies: list[Path] = []

    for unit, _, _ in iter_evaluate_specs():
        if unit.detector == ORACLE_DETECTOR or unit.detector == JOINT_HP_DETECTOR:
            continue

        directories.append(unit.logdir(evaluate.__name__))
        dependencies.append(unit.ocl_metrics)

    target = LOG_ROOT / "CLDD_B.parquet"
    return {
        "actions": [(collect_dataset, (directories, target))],
        "file_dep": dependencies + ["collect.py"],
        "targets": [target],
    }


def task_dd_collect():
    """Collect all dd_run outputs into one CSV for analysis."""

    dd_run_dirs: list[Path] = []
    file_deps: list[Path] = []

    for strategy, boundary, detector, (trial, _) in product(
        STRATEGY, BOUNDARY, DETECTOR, enumerate(EVALUATION_SEEDS)
    ):
        unit = Unit(strategy, detector, boundary, trial)
        dd_run_dirs.append(unit.logdir(dd_run.__name__))
        file_deps.append(unit.dd_run_metrics)

    target = LOG_ROOT / dd_run.__name__ / "data_frame.csv"
    output_html = LOG_ROOT / "profile" / "dd.html"
    return {
        "actions": [
            (collect_dd_run_records, (dd_run_dirs, target)),
            f"uv run script/profile.py 'Drift Detection Report' {target} {output_html}",
        ],
        "file_dep": file_deps + ["collect.py", "script/profile.py"],
        "targets": [target, output_html],
    }
