"""doit tasks for the drift-detection experiment pipeline.

This file implements the phase-oriented workflow described in ``plan.md``
for Experiment 1 (drift detector study):

1. Phase 1: strategy hyperparameter search with an oracle detector.
2. Phase 1b: write best hyperparameters back into strategy configs.
3. Phase 1c: evaluate tuned strategies to generate error-rate streams.
4. Phase 1d: combine per-seed detector metrics into one stream artifact.
5. Phase 2: detector hyperparameter search on the combined streams.
6. Phase 3: final multi-seed evaluation of tuned configs.
7. Phase 4: aggregate scalar metrics into a summary CSV.

The task graph is intentionally file-target driven so ``doit`` can skip work
that is already up to date.
"""
import json
import subprocess
from itertools import product
from pathlib import Path
from doit.task import Task
from dodo_helpers import (
    aggregate_metrics_csv,
    combine_dd_metrics,
    config_file,
    logdir,
    prepare_detector_config,
    select_best_phase2_detector,
    study_name,
)


STRATEGY = [
    "FT",
    "EWC",
]
BOUNDARY = [
    "00_abrupt",
    "01_gradual",
    "02_slow",
]
DETECTOR = [
    "ADWIN",
    "oracle",
]

VALID_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS = [10, 11, 12, 13, 14]

CMD = "uv run main.py -q"

def _task_name(prefix: str, *parts: object) -> str:
    """Build a stable doit subtask name."""
    return f"{prefix}_{'_'.join(str(part) for part in parts)}"


def _phase2_detectors() -> list[str]:
    """Return detectors that are actually tuned in phase 2."""
    return [detector for detector in DETECTOR if detector != "oracle"]


def _phase3_selection_path(label: str, strategy: str, boundary: str) -> Path:
    return logdir(label, strategy, boundary, "oracle", 0).parent / "selected_detector.json"


def _phase3_marker_path(label: str, strategy: str, boundary: str, seed: int) -> Path:
    return logdir(label, strategy, boundary, "oracle", seed).parent / f"{seed:03d}" / "phase3_eval_done.json"


def _phase3_selected_detector(strategy: str, boundary: str) -> str:
    """Resolve detector from the cached phase-2 selection artifact."""
    selection_path = _phase3_selection_path("final-eval", strategy, boundary)
    try:
        with selection_path.open() as f:
            payload = json.load(f)
        detector = payload.get("detector")
        if isinstance(detector, str) and detector:
            return detector
    except Exception:
        pass

    # Fallback keeps phase 3 runnable if selection artifact is missing.
    if "oracle" in DETECTOR:
        return "oracle"
    return DETECTOR[0]


def _phase3_uptodate(label: str, strategy: str, boundary: str, seed: int) -> bool:
    detector = _phase3_selected_detector(strategy, boundary)
    selection_path = _phase3_selection_path(label, strategy, boundary)
    marker = _phase3_marker_path(label, strategy, boundary, seed)
    detectors_to_check = ["oracle"]
    if detector != "oracle":
        detectors_to_check.append(detector)

    if not marker.exists() or not selection_path.exists():
        return False

    for detector_name in detectors_to_check:
        run_dir = logdir(label, strategy, boundary, detector_name, seed)
        dd_path = run_dir / "dd_metrics.pkl"
        ocl_path = run_dir / "ocl_metrics.pkl"
        if not dd_path.exists() or not ocl_path.exists():
            return False

    try:
        with marker.open() as f:
            payload = json.load(f)
    except Exception:
        return False

    return payload.get("detector") == detector


def _run_phase3_eval(label: str, strategy: str, boundary: str, seed: int) -> None:
    detector = _phase3_selected_detector(strategy, boundary)
    detectors_to_run = ["oracle"]
    if detector != "oracle":
        detectors_to_run.append(detector)

    for detector_name in detectors_to_run:
        cmd = (
            f"{CMD} -a label='{label}' -a seed={seed} -a trial={seed} "
            f"{config_file(strategy, boundary, detector_name)} run"
        )
        subprocess.run(cmd, shell=True, check=True)

    marker = _phase3_marker_path(label, strategy, boundary, seed)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("w") as f:
        json.dump({"detector": detector}, f)


def task_p2_select_detector():
    """Persist phase-2 detector choices to cached files for phase 3.

    This task captures Optuna study reads as explicit file targets so detector
    selection is cached and can participate in doit dependency tracking.
    """
    label = "final-eval"

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        target = _phase3_selection_path(label, strategy, boundary)
        name = _task_name("select_detector", strategy, boundary)
        detector_task_deps = [
            f"p2_hp_detector:{_task_name('hp_detector', strategy, boundary, detector)}"
            for detector in _phase2_detectors()
        ]

        file_dep = [
            Path(config_file(strategy, boundary)).as_posix(),
        ]
        file_dep.extend(
            [Path(config_file(strategy, boundary, detector)).as_posix() for detector in _phase2_detectors()]
        )

        yield {
            "name": name,
            "actions": [
                (
                    select_best_phase2_detector,
                    [strategy, boundary, _phase2_detectors(), target.as_posix()],
                )
            ],
            "targets": [target.as_posix()],
            "file_dep": file_dep,
            "task_dep": detector_task_deps,
        }

def task_p1_hp_strategy():
    """Phase 1: tune strategy hyperparameters with oracle boundaries.

    For each strategy-boundary pair, this runs the strategy hyperparameter
    search command and expects ``best_params.yaml`` as the materialized output.
    This corresponds to the plan's first phase where learner tuning is decoupled
    from detector imperfections by using an oracle detector.
    """

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        target = (
            logdir("hp", strategy, boundary, "oracle", 0).parent / "best_params.yaml"
        )
        name = _task_name("hp", strategy, boundary)
        yield {
            "name": name,
            "actions": [f"{CMD} {config_file(strategy, boundary)} hpsearch hp"],
            "targets": [target.as_posix()],
            "uptodate": [lambda: target.exists()],
        }


def task_p1_update_configs():
    """Phase 1b: inject best strategy hyperparameters into config files.

    The update step is only considered up to date when the destination config
    file exists and is at least as new as the generated ``best_params.yaml``.
    """

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        source = (
            logdir("hp", strategy, boundary, "oracle", 0).parent / "best_params.yaml"
        )
        target = Path(config_file(strategy, boundary))
        name = _task_name("update", strategy, boundary)
        yield {
            "name": name,
            "actions": [f"uv run update_hp.py '{study_name(strategy, boundary)}'"],
            "targets": [target.as_posix()],
            "file_dep": [source.as_posix()],
            "task_dep": [f"p1_hp_strategy:{_task_name('hp', strategy, boundary)}"],
            # Update the config file only if the best_params.yaml is newer than the
            # config file (i.e., if there are new hyperparameters to update).
            "uptodate": [
                lambda: (
                    source.exists()
                    and
                    target.exists()
                    and (target.stat().st_mtime >= source.stat().st_mtime)
                )
            ],
        }


def task_p1_eval():
    """Phase 1c: run per-seed error-stream evaluation for tuned strategies.

    These runs generate detector-facing metrics (``dd_metrics.pkl``) that are
    later reused for detector tuning, matching the plan's two-stage approach.
    """
    label = "error-stream"

    for strategy, boundary, eval_seed in product(STRATEGY, BOUNDARY, VALID_SEEDS):
        target = (
            logdir(label, strategy, boundary, "oracle", eval_seed) / "dd_metrics.pkl"
        )
        name = _task_name("eval", strategy, boundary, eval_seed)
        yield {
            "name": name,
            "actions": [
                f"{CMD} -a label='{label}' -a seed={eval_seed} -a trial={eval_seed} {config_file(strategy, boundary)} run"
            ],
            "targets": [target.as_posix()],
            "task_dep": [f"p1_update_configs:{_task_name('update', strategy, boundary)}"],
            "uptodate": [lambda: target.exists()],
        }


def task_p1_error_stream():
    """Phase 1d: combine per-seed error streams into one detector artifact.

    Produces a single ``dd_metrics.pkl`` per strategy-boundary pair, which is
    consumed by detector hyperparameter search in Phase 2.
    """
    label = "error-stream"

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        sources = [
            (logdir(label, strategy, boundary, "oracle", eval_seed) / "dd_metrics.pkl")
            for eval_seed in VALID_SEEDS
        ]
        target = (
            logdir(label, strategy, boundary, "oracle", 0).parent / "dd_metrics.pkl"
        )
        name = _task_name("error_stream", strategy, boundary)
        yield {
            "name": name,
            "actions": [
                (
                    combine_dd_metrics,
                    [[source.as_posix() for source in sources], target.as_posix()],
                )
            ],
            "file_dep": [source.as_posix() for source in sources],
            "task_dep": [
                f"p1_eval:{_task_name('eval', strategy, boundary, eval_seed)}"
                for eval_seed in VALID_SEEDS
            ],
            "targets": [target.as_posix()],
        }


def task_p2_hp_detector():
    """Phase 2: tune detector hyperparameters from recorded error streams.

    Detector tuning is executed only for non-oracle detectors. Each task:
    1. Materializes a detector-specific config.
    2. Runs detector hyperparameter search against combined phase-1 streams.
    3. Updates config files with best detector parameters.
    """
    label = "error-stream"

    for strategy, boundary, detector in product(STRATEGY, BOUNDARY, DETECTOR):
        if detector == "oracle":
            continue

        source = (
            logdir(label, strategy, boundary, "oracle", 0).parent / "dd_metrics.pkl"
        )
        detector_config = Path(config_file(strategy, boundary, detector))
        target = detector_config
        name = _task_name("hp_detector", strategy, boundary, detector)
        yield {
            "name": name,
            "actions": [
                (
                    prepare_detector_config,
                    [strategy, boundary, detector, detector_config.as_posix()],
                ),
                f"{CMD} {detector_config.as_posix()} dd_hpsearch {source.as_posix()}",
                f"uv run update_hp.py '{study_name(strategy, boundary, detector)}'",
            ],
            "file_dep": [
                source.as_posix(),
                Path(config_file(strategy, boundary)).as_posix(),
                (Path("config/base") / f"drift_detector/{detector}.yml").as_posix(),
            ],
            "task_dep": [
                f"p1_error_stream:{_task_name('error_stream', strategy, boundary)}",
                f"p1_update_configs:{_task_name('update', strategy, boundary)}",
            ],
            "targets": [target.as_posix()],
        }


def task_p3_final_evaluation():
    """Phase 3: final evaluation on held-out seeds.

    Runs oracle and tuned strategy-detector combinations on test seeds that are
    distinct from the hyperparameter-search seeds.

    Tasks are detector-agnostic at definition time and choose the detector at
    execution time: best phase-2 detector if available, otherwise oracle. The
    oracle run is always executed as a baseline.
    """
    label = "final-eval"

    for strategy, boundary in product(STRATEGY, BOUNDARY):
        for eval_seed in TEST_SEEDS:
            target = _phase3_marker_path(label, strategy, boundary, eval_seed)
            name = _task_name("final", strategy, boundary, eval_seed)
            detector_task_deps = [
                f"p2_select_detector:{_task_name('select_detector', strategy, boundary)}"
            ]
            if not detector_task_deps:
                detector_task_deps = [
                    f"p1_update_configs:{_task_name('update', strategy, boundary)}"
                ]
            yield {
                "name": name,
                "actions": [(_run_phase3_eval, [label, strategy, boundary, eval_seed])],
                "targets": [target.as_posix()],
                "file_dep": [_phase3_selection_path(label, strategy, boundary).as_posix()],
                "task_dep": detector_task_deps,
                "uptodate": [
                    lambda strategy=strategy, boundary=boundary, eval_seed=eval_seed: _phase3_uptodate(
                        label, strategy, boundary, eval_seed
                    )
                ],
            }


def task_p4_aggregate_metrics():
    """Phase 4: aggregate final scalar metrics into ``logs/final-eval/metrics.csv``.

    This consolidates detector and continual-learning metrics from all final
    evaluation runs into a single table for downstream analysis.
    """
    label = "final-eval"
    target = Path("logs") / label / "metrics.csv"

    file_dep = []
    for strategy, boundary in product(STRATEGY, BOUNDARY):
        for seed in TEST_SEEDS:
            file_dep.append(_phase3_marker_path(label, strategy, boundary, seed).as_posix())

    yield {
        "name": "metrics_final_eval",
        "actions": [
            (
                aggregate_metrics_csv,
                [label, target.as_posix(), STRATEGY, BOUNDARY, DETECTOR, TEST_SEEDS],
            )
        ],
        "task_dep": ["p3_final_evaluation"],
        "file_dep": file_dep,
        "targets": [target.as_posix()],
    }
