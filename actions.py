from pathlib import Path
from typing import Sequence, Dict, Any, List
from loguru import logger
from subprocess import check_call as _call, CalledProcessError, STDOUT
from omegaconf import OmegaConf


def call(
    command: str,
    configs: Sequence[Path],
    *args: Any,
    dotlist: Dict[str, Any] | None = None,
    identifier: str | None = None,
):
    if dotlist is None:
        dotlist = {}

    # Disable progress bar for all subprocesses to avoid messy logs. Can be re-enabled
    # with config overrides.
    dotlist["disable_progress_bar"] = True

    # Construct command with config files and dotlist overrides
    cmd = "uv run main.py".split(" ")
    for config in configs:
        cmd += ["-c", config.as_posix()]
    for k, v in dotlist.items():
        cmd += ["-a", f"{k}={v}"]
    cmd += [command]
    cmd += map(str, args)

    log_file = None
    if identifier is not None:
        log_file = Path("logs") / "console" / f"{identifier}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if log_file is None:
            _call(cmd)
        else:
            with open(log_file, "w") as f:
                f.write(f"$ {' '.join(cmd)}\n\n")
                _call(cmd, stdout=f, stderr=STDOUT)
    except CalledProcessError:
        logger.error(f"error occured check logs at: {log_file}")
        exit(1)


def tune_strategy(configs: Sequence[Path], identifier: str):
    call("hpsearch", configs, tune_strategy.__name__, identifier=identifier)


def error_stream(
    configs: Sequence[Path],
    seed: int,
    trial: int,
    identifier: str,
):
    label = error_stream.__name__
    call(
        "run",
        configs,
        dotlist={
            "seed": seed,
            "trial": trial,
            "label": label,
        },
        identifier=f"{label}.{identifier}",
    )


def evaluate(
    configs: Sequence[Path],
    seed: int,
    trial: int,
    detector_label: str,
    identifier: str,
):
    label = evaluate.__name__
    dotlist = {
        "seed": seed,
        "trial": trial,
        "label": label,
        "drift_detector.label": detector_label,
    }
    call("run", configs, dotlist=dotlist, identifier=f"{label}.{identifier}")


def tune_detector(
    configs: Sequence[Path],
    error_streams: List[Path],
    identifier: str,
):
    label = tune_detector.__name__
    call(
        "dd_hpsearch",
        configs,
        *error_streams,
        dotlist={"label": label, "hpsearch.n_trials": 30},
        identifier=f"{label}.{identifier}",
    )


def select_best_detector(
    trial_files: List[Path],
    target: Path,
):
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


def collect_evaluate_records(
    evaluate_dirs: List[Path],
    target: Path,
):
    from collect import load_records

    target.parent.mkdir(parents=True, exist_ok=True)
    df = load_records(evaluate_dirs)
    df.to_csv(target, index=False, float_format="%.4f")
