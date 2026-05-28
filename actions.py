from pathlib import Path
from typing import Sequence, Dict, Any, List
from loguru import logger
from subprocess import check_call as _call, CalledProcessError
from omegaconf import OmegaConf


def call(
    command: str, configs: Sequence[Path], *args: Any, dotlist: Dict[str, Any] = {}
):
    cmd = "uv run main.py".split(" ")
    for config in configs:
        cmd += ["-c", config.as_posix()]
    for k, v in dotlist.items():
        cmd += ["-a", f"{k}={v}"]
    cmd += [command]
    cmd += map(str, args)
    logger.info(" ".join(cmd))
    try:
        _call(cmd)
    except CalledProcessError:
        exit(1)


def tune_strategy(configs: Sequence[Path]):
    call("hpsearch", configs, tune_strategy.__name__)


def error_stream(
    configs: Sequence[Path],
    seed: int,
    trial: int,
):
    call(
        "run",
        configs,
        dotlist={"seed": seed, "trial": trial, "label": error_stream.__name__},
    )


def evaluate(
    configs: Sequence[Path],
    seed: int,
    trial: int,
    detector_label: str,
):
    call(
        "run",
        configs,
        dotlist={
            "seed": seed,
            "trial": trial,
            "label": evaluate.__name__,
            "drift_detector.label": detector_label,
        },
    )


def tune_detector(
    configs: Sequence[Path],
    error_streams: List[Path],
):
    call(
        "dd_hpsearch",
        configs,
        *error_streams,
        dotlist={"label": tune_detector.__name__},
    )


def select_best_detector(
    trial_files: List[Path],
    target: Path,
):
    logger.info(f"Selecting best detector from trials: {trial_files}")
    target.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    for trial_file in trial_files:
        with open(trial_file) as f:
            trial_dict = OmegaConf.load(f)
            trials.append(trial_dict)
    best_trial = max(trials, key=lambda t: t["value"])
    selection = {
        "drift_detector": {
            "type_": best_trial["config"]["drift_detector"]["type_"],
            **best_trial["params"]["drift_detector"],
        }
    }
    with open(target, "w") as f:
        f.write(f"# Selected best trial with score {best_trial['value']}\n")
        f.write(OmegaConf.to_yaml(selection))
