from itertools import product
from pathlib import Path

STRATEGY = [
    "FT",
    # "EWC",
]
BOUNDARY = [
    "00_abrupt",
    # "01_gradual",
    # "02_slow",
]
DETECTOR = [
    "ADWIN",
]
HP_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS = [5, 6, 7, 8, 9]
HP_TRIALS = 5


def config_file(strategy, boundary, detector=None):
    if detector is None:
        return f"config/{boundary}/{strategy}_oracle_MLP.yml"
    else:
        return f"config/{boundary}/{strategy}_{detector}_{boundary}_MLP.yml"


def study_name(strategy, boundary, detector=None):
    if detector is None:
        return f"bocl/hp/{boundary}/{strategy}_oracle_MLP"
    else:
        return f"bocl/hp/{boundary}/{strategy}_{detector}_{boundary}_MLP"


def logdir(label: str, strategy: str, boundary: str, detector: str, trial: int) -> Path:
    # logs/hp/00_abrupt/FT_oracle_MLP/000
    return Path(f"logs/{label}/{boundary}/{strategy}_{detector}_MLP/{trial:03d}")


def dir_exists(paths: list[str]) -> bool:
    return all(Path(path).exists() for path in paths)


CMD = "uv run main.py"


def task_p1_hp():
    for strategy, boundary in product(STRATEGY, BOUNDARY):
        target = (
            logdir("hp", strategy, boundary, "oracle", 0).parent / "best_params.yaml"
        )
        yield {
            "name": f"hp_{strategy}_{boundary}",
            "actions": [f"{CMD} {config_file(strategy, boundary)} hpsearch hp"],
            "targets": [target.as_posix()],
            "uptodate": [lambda: target.exists()],
        }


def task_p1_update_configs():
    for strategy, boundary in product(STRATEGY, BOUNDARY):
        source = (
            logdir("hp", strategy, boundary, "oracle", 0).parent / "best_params.yaml"
        )
        target = Path(config_file(strategy, boundary))
        yield {
            "name": f"update_{strategy}_{boundary}",
            "actions": [f"uv run update_hp.py '{study_name(strategy, boundary)}'"],
            "targets": [target.as_posix()],
            "file_dep": [source.as_posix()],
            # Update the config file only if the best_params.yaml is newer than the
            # config file (i.e., if there are new hyperparameters to update).
            "uptodate": [
                lambda: (
                    target.exists()
                    and (target.stat().st_mtime >= source.stat().st_mtime)
                )
            ],
        }


def task_p1_eval():
    label = "error-stream"

    for strategy, boundary, eval_seed in product(STRATEGY, BOUNDARY, EVAL_SEEDS):
        target = (
            logdir(label, strategy, boundary, "oracle", eval_seed) / "dd_metrics.pkl"
        )
        yield {
            "name": f"eval_{strategy}_{boundary}_{eval_seed}",
            "actions": [
                f"{CMD} -a label='{label}' -a seed={eval_seed} -a trial={eval_seed} {config_file(strategy, boundary)} run"
            ],
            "targets": [target.as_posix()],
            "uptodate": [lambda: target.exists()],
        }
