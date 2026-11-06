"""Pick the best drift detector for one strategy and boundary from phase-2 results.

The rule is fixed before any evaluation run:

1. Highest mean `my_f1` over the tuning error streams.
2. Ties go to the lower mean detection time (`mdt`). All detectors share the same
   detection window, so this is the same as lower NMDT.
3. Remaining ties follow the order of the detectors given.

A detector that never fires on any tuning stream ranks last.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import OmegaConf


def rank_detectors(
    scores: Mapping[str, Mapping[str, float]], order: Sequence[str]
) -> list[str]:
    """Return detectors from best to worst.

    :param scores: Mean phase-2 metrics per detector, with keys `my_f1`, `mdt` and
        `n_alarms`.
    :param order: Tie-break order. Must hold every key of `scores`.
    """

    def key(detector: str):
        metrics = scores[detector]
        silent = metrics["n_alarms"] == 0
        mdt = metrics["mdt"]
        if mdt is None or math.isnan(mdt):
            mdt = math.inf
        return (silent, -metrics["my_f1"], mdt, order.index(detector))

    return sorted(scores, key=key)


def write_best_detector(
    best_trials: Mapping[str, Path],
    detector_configs: Mapping[str, Path],
    order: Sequence[str],
    label: str,
    target: Path,
):
    """Write a config with the drift detector of the best phase-2 trial.

    :param best_trials: `best_trial.yml` from phase 2, per detector.
    :param detector_configs: `config/detector/<name>.yml`, per detector.
    :param order: Tie-break order.
    :param label: Detector label of the evaluation runs.
    :param target: Output file. Its `selection` key holds the ranking, which the
        config loader ignores.
    """
    trials = {name: OmegaConf.load(path) for name, path in best_trials.items()}
    scores = {
        name: {
            k: float(trial.user_attrs.dd_metrics[k])
            for k in ("my_f1", "mdt", "n_alarms")
        }
        for name, trial in trials.items()
    }
    ranking = rank_detectors(scores, order)
    best = ranking[0]

    detector = OmegaConf.merge(
        OmegaConf.load(detector_configs[best]).drift_detector,
        trials[best].params.drift_detector,
        {"label": label},
    )
    out = OmegaConf.create(
        {
            "drift_detector": detector,
            "selection": {
                "best": best,
                "ranking": [{"detector": d, **scores[d]} for d in ranking],
            },
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(out, target)
