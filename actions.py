import itertools
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from subprocess import STDOUT, CalledProcessError
from subprocess import check_call as _call
from typing import Any

from loguru import logger

_GPUS = [g for g in os.environ.get("CLDD_GPUS", "").split(",") if g]
_next_gpu = itertools.count()
_gpu_lock = threading.Lock()


def _gpu_env() -> dict | None:
    """Give each subprocess the next GPU from `CLDD_GPUS` (e.g. "0,1"), in turn."""
    if not _GPUS:
        return None
    with _gpu_lock:
        gpu = _GPUS[next(_next_gpu) % len(_GPUS)]
    return {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}


def call(
    command: str,
    configs: Sequence[Path],
    *args: Any,
    dotlist: dict[str, Any] | None = None,
    identifier: str | None = None,
):
    if dotlist is None:
        dotlist = {}

    # Disable progress bar for all subprocesses to avoid messy logs. Can be re-enabled
    # with config overrides.
    dotlist["disable_progress_bar"] = True

    # Construct command with config files and dotlist overrides
    cmd = ["uv", "run", "main.py"]
    for config in configs:
        cmd += ["-c", config.as_posix()]
    for k, v in dotlist.items():
        cmd += ["-a", f"{k}={v}"]
    cmd += [command]
    cmd += map(str, args)

    log_file = None
    if identifier is not None:
        log_file = (
            Path(os.environ.get("LOGDIR", "logs"))
            / "cldd"
            / "console"
            / f"{identifier}.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        env = _gpu_env()
        if log_file is None:
            _call(cmd, env=env)
        else:
            with open(log_file, "w") as f:
                gpu = (
                    f" (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})"
                    if env
                    else ""
                )
                f.write(f"$ {' '.join(cmd)}{gpu}\n\n")
                _call(cmd, stdout=f, stderr=STDOUT, env=env)
    except CalledProcessError:
        logger.error(f"error occured check logs at: {log_file}")
        sys.exit(1)


def tune_strategy(configs: Sequence[Path], identifier: str):
    call(
        "hpsearch",
        configs,
        tune_strategy.__name__,
        identifier=identifier,
        dotlist={"scenario.validation": True},
    )


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
    error_streams: list[Path],
    identifier: str,
):
    label = tune_detector.__name__
    call(
        "dd_hpsearch",
        configs,
        *error_streams,
        dotlist={"label": label},
        identifier=f"{label}.{identifier}",
    )


def dd_run(
    configs: Sequence[Path],
    error_stream: Path,
    trial: int,
    detector_label: str,
    identifier: str,
):
    label = dd_run.__name__
    call(
        "dd_run",
        configs,
        error_stream,
        dotlist={
            "label": label,
            "trial": trial,
            "drift_detector.label": detector_label,
        },
        identifier=f"{label}.{identifier}",
    )


def collect_evaluate_records(
    evaluate_dirs: list[Path],
    target: Path,
):
    from collect import load_records

    target.parent.mkdir(parents=True, exist_ok=True)
    df = load_records(evaluate_dirs)
    df.to_csv(target, index=False, float_format="%.4f")


def collect_dd_run_records(
    dd_run_dirs: list[Path],
    target: Path,
):
    from collect import load_dd_run_records

    target.parent.mkdir(parents=True, exist_ok=True)
    df = load_dd_run_records(dd_run_dirs)
    df.to_csv(target, index=False, float_format="%.4f")


def select_best(
    best_trials: dict[str, Path],
    detector_configs: dict[str, Path],
    order: list[str],
    label: str,
    target: Path,
):
    from src.select_best import write_best_detector

    write_best_detector(best_trials, detector_configs, order, label, target)
