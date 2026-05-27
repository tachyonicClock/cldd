from itertools import product
from numbers import Number
from pathlib import Path
import csv
import json
import pickle
import os
import optuna
import yaml


def config_file(strategy, boundary, detector=None):
    if detector is None:
        return f"config/{boundary}/{strategy}_oracle_MLP.yml"
    else:
        return f"config/{boundary}/{strategy}_{detector}_MLP.yml"


def study_name(strategy, boundary, detector=None):
    if detector is None:
        return f"bocl/hp/{boundary}/{strategy}_oracle_MLP"
    else:
        return f"bocl/dd_hpsearch/{boundary}/{strategy}_{detector}_MLP"


def logdir(label: str, strategy: str, boundary: str, detector: str, trial: int) -> Path:
    # logs/hp/00_abrupt/FT_oracle_MLP/000
    return Path(f"logs/{label}/{boundary}/{strategy}_{detector}_MLP/{trial:03d}")


def dir_exists(paths: list[str]) -> bool:
    return all(Path(path).exists() for path in paths)


def combine_dd_metrics(sources: list[str], target: str) -> None:
    combined = []
    for source in sources:
        with Path(source).open("rb") as f:
            combined.append(pickle.load(f))

    with Path(target).open("wb") as f:
        pickle.dump(combined, f)


def prepare_detector_config(
    strategy: str, boundary: str, detector: str, target: str
) -> None:
    with Path(config_file(strategy, boundary)).open() as f:
        config = yaml.safe_load(f) or {}

    config["bases"] = [
        "scenario.yml",
        f"boundary/{boundary}.yml",
        f"learner/{strategy}.yml",
        f"drift_detector/{detector}.yml",
    ]

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def select_best_phase2_detector(
    strategy: str,
    boundary: str,
    detectors: list[str],
    target: str,
) -> None:
    """Persist the best phase-2 detector choice for one strategy/boundary pair."""
    storage = os.environ.get("OPTUNA_STORAGE")

    best_detector = None
    best_value = None
    for detector in detectors:
        try:
            study = optuna.load_study(
                study_name=study_name(strategy, boundary, detector),
                storage=storage,
            )
        except Exception:
            continue

        value = float(study.best_trial.value)
        if best_value is None or value > best_value:
            best_value = value
            best_detector = detector

    # Keep phase 3 runnable even if no detector studies are available yet.
    if best_detector is None:
        best_detector = "oracle"

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w") as f:
        json.dump(
            {
                "strategy": strategy,
                "boundary": boundary,
                "detector": best_detector,
                "best_value": best_value,
            },
            f,
        )


def _scalar_value(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, Number) or isinstance(value, bool):
        return value
    return None


def _extract_scalar_metrics(metrics, prefix: str) -> dict[str, Number | bool]:
    if isinstance(metrics, dict):
        items = metrics.items()
    elif hasattr(metrics, "__dict__"):
        items = metrics.__dict__.items()
    else:
        return {}

    scalars = {}
    for key, value in items:
        scalar = _scalar_value(value)
        if scalar is not None:
            scalars[f"{prefix}{key}"] = scalar
    return scalars


def aggregate_metrics_csv(
    label: str,
    target: str,
    strategies: list[str],
    boundaries: list[str],
    detectors: list[str],
    seeds: list[int],
) -> None:
    rows = []
    for strategy, boundary, seed in product(strategies, boundaries, seeds):
        boundary_dir = Path("logs") / label / boundary
        discovered_detectors = []
        if boundary_dir.exists():
            prefix = f"{strategy}_"
            suffix = "_MLP"
            for entry in boundary_dir.iterdir():
                if not entry.is_dir():
                    continue
                name = entry.name
                if name.startswith(prefix) and name.endswith(suffix):
                    discovered_detectors.append(name[len(prefix) : -len(suffix)])

        # Prefer the detector explicitly selected during phase 3 when available.
        marker = (
            logdir(label, strategy, boundary, "oracle", seed).parent
            / f"{seed:03d}"
            / "selected_detector.json"
        )
        selected_detector = None
        if marker.exists():
            try:
                with marker.open() as f:
                    payload = json.load(f)
                detector = payload.get("detector")
                if isinstance(detector, str) and detector:
                    selected_detector = detector
            except Exception:
                pass

        detector_candidates = list(
            dict.fromkeys([*detectors, "oracle", *discovered_detectors])
        )
        if selected_detector is not None and selected_detector not in detector_candidates:
            detector_candidates = [selected_detector, *detector_candidates]

        for detector in detector_candidates:
            run_dir = logdir(label, strategy, boundary, detector, seed)
            dd_path = run_dir / "dd_metrics.pkl"
            ocl_path = run_dir / "ocl_metrics.pkl"
            if not dd_path.exists() or not ocl_path.exists():
                continue

            with dd_path.open("rb") as f:
                dd_metrics = pickle.load(f)
            with ocl_path.open("rb") as f:
                ocl_metrics = pickle.load(f)

            row = {
                "label": label,
                "strategy": strategy,
                "boundary": boundary,
                "detector": detector,
                "seed": seed,
                "trial": seed,
                "run_dir": run_dir.as_posix(),
            }
            row.update(_extract_scalar_metrics(dd_metrics, "dd_"))
            row.update(_extract_scalar_metrics(ocl_metrics, "ocl_"))
            rows.append(row)

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with target_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "label",
                    "strategy",
                    "boundary",
                    "detector",
                    "seed",
                    "trial",
                    "run_dir",
                ],
            )
            writer.writeheader()
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with target_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
