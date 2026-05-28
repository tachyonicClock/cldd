from itertools import product
from pathlib import Path
from typing import Sequence
from types_ import Strategy, Detector, Boundary
from actions import tune_strategy_hp, error_streams


STRATEGY = ["EWC", "FT"]
DETECTOR = ["ADWIN", "ORACLE"]
BOUNDARY = [
    "abrupt",
    # "gradual",
    # "slow",
]
SEEDS = [0, 1]
ROOT = Path("doitdata")


def get_targets(tasks):
    targets = []
    for task in tasks:
        targets.extend(task["targets"])
    return targets


def _target(
    func, strategy: Strategy, detector: Detector, boundary: Boundary, seed: int = 0
) -> Path:
    id_ = get_identifier(strategy, detector, boundary, seed)
    target = ROOT / func.__name__ / f"{id_}.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def get_identifier(
    strategy: Strategy, detector: Detector, boundary: Boundary, seed: int = 0
) -> str:
    return f"{strategy}.{detector}.{boundary}.{seed:02d}"


def split_identifier(id_: str) -> tuple[Strategy, Detector, Boundary, int]:
    strategy, detector, boundary, seed = id_.split(".")
    return strategy, detector, boundary, int(seed)


def task_tune_strategy_hp():
    detector = "ORACLE"
    for strategy, boundary in product(STRATEGY, BOUNDARY):
        id_ = get_identifier(strategy, detector, boundary)

        target = _target(task_tune_strategy_hp, strategy, detector, boundary)

        args = dict(
            strategy=strategy,
            detector=detector,
            boundary=boundary,
        )
        yield {
            "name": id_,
            "actions": [(tune_strategy_hp, (), args)],
            "file_dep": [],
            "targets": [target],
        }


# def task_error_streams():
#     for strategy, detector, boundary, seed in product(STRATEGY, DETECTOR, BOUNDARY, SEEDS):
#         id_ = get_identifier(strategy, detector, boundary, seed)
#         strategy_config = _target(task_tune_strategy_hp, strategy, "ORACLE", boundary)
#         target = _target(task_error_streams, strategy, detector, boundary, seed)
#         args = [target, strategy_config, strategy, detector, boundary, seed]
#         yield {
#             "name": id_,
#             "actions": [(error_streams, args)],
#             "file_dep": [strategy_config],
#             "targets": [target],
#         }


# def task_tune_hp_detector():
#     root = ROOT / task_tune_hp_detector.__name__
#     root.mkdir(parents=True, exist_ok=True)

#     # Group error streams by strategy and boundary, so that we can tune the detector for
#     # each group.
#     grouped_tasks = {}
#     for task in task_error_streams():
#         strategy, detector, boundary, seed = split_identifier(task["name"])
#         grouped_tasks.setdefault((strategy, boundary), []).append(task)

#     # For each group, tune the detector and create a target file.
#     for (strategy, boundary), tasks in grouped_tasks.items():
#         for detector in DETECTOR:
#             id_ = get_identifier(strategy, detector, boundary)
#             target = _target(task_tune_hp_detector, strategy, detector, boundary)
#             yield {
#                 "name": id_,
#                 "actions": [f"touch {target}"],
#                 "file_dep": get_targets(tasks),
#                 "targets": [target],
#             }


# def task_select_best_detector():

#     # Group task_tune_hp_detector by strategy and boundary, so that we can find the best
#     # detector for each group.
#     grouped_tasks = {}
#     for task in task_tune_hp_detector():
#         strategy, _, boundary, _ = split_identifier(task["name"])
#         grouped_tasks.setdefault((strategy, boundary), []).append(task)

#     # For each group, find the best detector and create a target file.
#     for (strategy, boundary), tasks in grouped_tasks.items():
#         id_ = get_identifier(strategy, "BEST", boundary)
#         target = _target(task_select_best_detector, strategy, "BEST", boundary)
#         yield {
#             "name": id_,
#             "actions": [f"touch {target}"],
#             "file_dep": get_targets(tasks),
#             "targets": [target],
#         }


# def task_evaluate_oracle_detector():
#     for task in task_tune_strategy_hp():
#         strategy, detector, boundary, seed = split_identifier(task["name"])
#         assert detector == "ORACLE"
#         id_ = get_identifier(strategy, "ORACLE", boundary, seed)

#         target = _target(
#             task_evaluate_oracle_detector, strategy, "ORACLE", boundary, seed
#         )
#         yield {
#             "name": id_,
#             "actions": [f"touch {target}"],
#             "file_dep": task["targets"],
#             "targets": [target],
#         }


# def task_evaluate_best_detector():
#     root = ROOT / task_evaluate_best_detector.__name__
#     root.mkdir(parents=True, exist_ok=True)

#     for strategy, boundary in product(STRATEGY, BOUNDARY):
#         id_ = get_identifier(strategy, "BEST", boundary)
#         target          = _target(task_evaluate_best_detector, strategy, "BEST", boundary)
#         detector_config = _target(task_select_best_detector, strategy, "BEST", boundary)
#         strategy_config = _target(task_tune_strategy_hp, strategy, "ORACLE", boundary)

#         yield {
#             "name": id_,
#             "actions": [f"touch {target}"],
#             "file_dep": [detector_config, strategy_config],
#             "targets": [target],
#         }
