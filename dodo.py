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
    select_best,
    tune_strategy,
    error_stream,
    tune_detector,
    dd_run,
    evaluate,
    collect_evaluate_records,
    collect_dd_run_records,
)
from collect import collect_dataset
from src.config import DEFAULT_DATASET_LABEL
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
    "SEED_ONE_SIDED",
    "STEPD_ONE_SIDED",
]
BOUNDARY = [
    "abrupt",
    "gradual",
    "slow",
]

JOINT_HP_DETECTOR = "ADWIN_JOINT"
"""The detector used for joint tuning of strategy and detector HP."""

BEST_DETECTOR = "BEST"
"""Label of runs that use the detector picked by `task_select_best`."""

DEFAULT_DATASET = DEFAULT_DATASET_LABEL
"""Main case study (RotatedFashionMNIST). Its log paths have no dataset level."""
CHECK_DATASETS = ["dcifar100"]
"""Datasets for the smaller check. Phase 3 runs only the oracle, the fixed detector and
the best detector. Each has a `config/dataset/<name>.yml`."""
FIXED_DETECTOR = "ADWIN"
"""Detector picked on the main dataset, run live on check datasets with its phase-2
HP from that dataset."""
DATASETS = [DEFAULT_DATASET, *CHECK_DATASETS]

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

LOG_ROOT = Path(os.environ.get("LOGDIR", "logs")) / "cldd"


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
    dataset: str = DEFAULT_DATASET

    @property
    def trial_str(self) -> str:
        return f"{self.trial:03d}"

    @property
    def is_default_dataset(self) -> bool:
        return self.dataset == DEFAULT_DATASET

    @property
    def identifier(self) -> str:
        _id_path = [self.strategy, self.detector, self.boundary, self.trial_str]
        if not self.is_default_dataset:
            _id_path.insert(0, self.dataset)
        return ".".join(_id_path)

    def configs(self, hp: str | None = None) -> list[Path]:
        r = Path("config")
        configs = [
            r / "base.yml",
            r / "strategy" / f"{self.strategy}.yml",
            r / "detector" / f"{self.detector}.yml",
            r / "boundary" / f"{self.boundary}.yml",
        ]
        if not self.is_default_dataset:
            configs.append(r / "dataset" / f"{self.dataset}.yml")
        if hp is not None:
            configs.append(r / "hp" / f"{hp}.yml")
        if DEBUG_MODE:
            configs.append(r / "debug.yml")
        return configs

    def logdir(self, label: str) -> Path:
        root = LOG_ROOT if self.is_default_dataset else LOG_ROOT / self.dataset
        return (
            root
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
    def best_detector(self) -> Path:
        return self.logdir(select_best.__name__).parent / "best_detector.yml"

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

    def task_select_best(self) -> dict:
        best_trials = {
            d: Unit(
                self.strategy, d, self.boundary, dataset=self.dataset
            ).tune_detector_metrics
            for d in DETECTOR
        }
        detector_configs = {
            d: Path("config") / "detector" / f"{d}.yml" for d in DETECTOR
        }
        return {
            "name": self.identifier,
            "actions": [
                (
                    select_best,
                    (
                        best_trials,
                        detector_configs,
                        DETECTOR,
                        BEST_DETECTOR,
                        self.best_detector,
                    ),
                )
            ],
            "file_dep": [*best_trials.values(), *detector_configs.values()],
            "targets": [self.best_detector],
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

    for dataset, strategy, boundary in product(DATASETS, STRATEGY, BOUNDARY):
        yield Unit(
            strategy, ORACLE_DETECTOR, boundary, dataset=dataset
        ).task_tune_strategy()


def task_tune_detector_strategy():
    for strategy, boundary in product(STRATEGY, BOUNDARY):
        if strategy in DETECTOR_AGNOSTIC:
            continue
        yield Unit(strategy, JOINT_HP_DETECTOR, boundary).task_tune_strategy()


def task_error_stream():
    """Phase 1 output: create per-seed error streams using tuned strategy HP."""

    for dataset, strategy, boundary, (trial, seed) in product(
        DATASETS, STRATEGY, BOUNDARY, enumerate(ERROR_STREAM_SEEDS)
    ):
        yield Unit(
            strategy, ORACLE_DETECTOR, boundary, trial, dataset
        ).task_error_stream(seed)


def task_tune_hp_detector():
    """Phase 2: tune detector HP from all error streams of a strategy/boundary."""

    for dataset, strategy, boundary, detector in product(
        DATASETS, STRATEGY, BOUNDARY, DETECTOR
    ):
        # Collect error streams for all trials of this strategy and boundary, which will
        # be used for tuning the detector.
        error_streams = []
        for trial, _ in enumerate(ERROR_STREAM_SEEDS):
            error_streams.append(
                Unit(strategy, ORACLE_DETECTOR, boundary, trial, dataset).error_stream
            )

        yield Unit(strategy, detector, boundary, dataset=dataset).task_tune_hp_detector(
            error_streams
        )


def task_select_best():
    """Pick the best phase-2 detector for each strategy/boundary (see src/select_best.py)."""

    for dataset, strategy, boundary in product(DATASETS, STRATEGY, BOUNDARY):
        if strategy in DETECTOR_AGNOSTIC:
            continue
        yield Unit(
            strategy, BEST_DETECTOR, boundary, dataset=dataset
        ).task_select_best()


def iter_evaluate_specs(dataset: str = DEFAULT_DATASET):
    """Yield evaluate unit, seed, and extra config dependencies."""

    for strategy, boundary, (trial, seed) in product(
        STRATEGY, BOUNDARY, enumerate(EVALUATION_SEEDS)
    ):
        strategy_hp = Unit(
            strategy, ORACLE_DETECTOR, boundary, dataset=dataset
        ).tune_strategy_hp

        yield (
            Unit(strategy, ORACLE_DETECTOR, boundary, trial, dataset),
            seed,
            [strategy_hp],
        )
        if strategy in DETECTOR_AGNOSTIC:
            continue

        if dataset != DEFAULT_DATASET:
            fixed_hp = Unit(
                strategy, FIXED_DETECTOR, boundary, dataset=dataset
            ).tune_detector_hp
            yield (
                Unit(strategy, FIXED_DETECTOR, boundary, trial, dataset),
                seed,
                [strategy_hp, fixed_hp],
            )
            best_hp = Unit(strategy, BEST_DETECTOR, boundary, dataset=dataset)
            yield (
                Unit(strategy, BEST_DETECTOR, boundary, trial, dataset),
                seed,
                [strategy_hp, best_hp.best_detector],
            )
            continue

        for detector in DETECTOR:
            detector_hp = Unit(strategy, detector, boundary).tune_detector_hp
            yield (
                Unit(strategy, detector, boundary, trial),
                seed,
                [strategy_hp, detector_hp],
            )

        joint_detector_hp = Unit(strategy, JOINT_HP_DETECTOR, boundary).tune_strategy_hp
        yield (
            Unit(strategy, JOINT_HP_DETECTOR, boundary, trial),
            seed,
            [joint_detector_hp],
        )


def task_evaluate():
    """Phase 3: evaluate tuned configurations on held-out evaluation seeds.

    Detector-agnostic strategies are evaluated once with the oracle boundary
    setting. On the main dataset, boundary-aware strategies are evaluated with every
    detector. On check datasets, only with the fixed and the best detector.
    """

    for dataset in DATASETS:
        for unit, seed, configs in iter_evaluate_specs(dataset):
            yield unit.task_evaluate(seed, configs)


def iter_dd_run_units():
    """Yield units that replay each tuned detector on an oracle evaluate stream."""

    for dataset, strategy, boundary, detector, (trial, _) in product(
        DATASETS, STRATEGY, BOUNDARY, DETECTOR, enumerate(EVALUATION_SEEDS)
    ):
        yield Unit(strategy, detector, boundary, trial, dataset)


def task_dd_run():
    """Replay tuned detector configs on oracle evaluate error streams."""

    for unit in iter_dd_run_units():
        detector_hp = Unit(
            unit.strategy, unit.detector, unit.boundary, dataset=unit.dataset
        ).tune_detector_hp
        oracle_metrics = Unit(
            unit.strategy, ORACLE_DETECTOR, unit.boundary, unit.trial, unit.dataset
        ).evaluate_dd_metrics
        yield unit.task_dd_run(oracle_metrics, [detector_hp])


def dataset_root(dataset: str) -> Path:
    return LOG_ROOT if dataset == DEFAULT_DATASET else LOG_ROOT / dataset


def task_collect_evaluate():
    """Collect all evaluate outputs into one CSV per dataset for analysis."""

    for dataset in DATASETS:
        evaluate_dirs: list[Path] = []
        file_deps: list[Path] = []

        for unit, _, _ in iter_evaluate_specs(dataset):
            evaluate_dirs.append(unit.logdir(evaluate.__name__))
            file_deps.append(unit.ocl_metrics)

        root = dataset_root(dataset)
        target = root / evaluate.__name__ / "data_frame.csv"
        output_html = root / "profile" / "evaluate.html"
        yield {
            "name": dataset,
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
    a continual learning algorithm. It holds every dataset, told apart by its
    `dataset` column.
    """
    directories: list[Path] = []
    dependencies: list[Path] = []

    for dataset in DATASETS:
        for unit, _, _ in iter_evaluate_specs(dataset):
            if unit.detector != ORACLE_DETECTOR:
                continue

            directories.append(unit.logdir(evaluate.__name__))
            dependencies.append(unit.ocl_metrics)

        for strategy, boundary in product(STRATEGY, BOUNDARY):
            # Collect error streams for all trials of this strategy and boundary, which
            # will be used for tuning the detector.
            for trial, _ in enumerate(ERROR_STREAM_SEEDS):
                unit = Unit(strategy, ORACLE_DETECTOR, boundary, trial, dataset)
                directories.append(unit.error_stream.parent)
                dependencies.append(unit.error_stream)

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
    continual learning algorithm. Main dataset only."""
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
    """Collect all dd_run outputs into one CSV per dataset for analysis."""

    for dataset in DATASETS:
        units = [u for u in iter_dd_run_units() if u.dataset == dataset]
        dd_run_dirs = [u.logdir(dd_run.__name__) for u in units]
        file_deps = [u.dd_run_metrics for u in units]

        root = dataset_root(dataset)
        target = root / dd_run.__name__ / "data_frame.csv"
        output_html = root / "profile" / "dd.html"
        yield {
            "name": dataset,
            "actions": [
                (collect_dd_run_records, (dd_run_dirs, target)),
                f"uv run script/profile.py 'Drift Detection Report' {target} {output_html}",
            ],
            "file_dep": file_deps + ["collect.py", "script/profile.py"],
            "targets": [target, output_html],
        }
