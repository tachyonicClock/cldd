import math
from omegaconf import OmegaConf
from src.config import get_config
from src.select_best import rank_detectors, write_best_detector

ORDER = ["ADWIN", "DDM", "PH"]


def score(f1, mdt, alarms=4):
    return {"my_f1": f1, "mdt": mdt, "n_alarms": alarms}


def test_highest_f1_wins():
    scores = {"ADWIN": score(0.7, 10), "DDM": score(0.9, 500), "PH": score(0.8, 1)}
    assert rank_detectors(scores, ORDER) == ["DDM", "PH", "ADWIN"]


def test_tie_goes_to_lower_mdt_then_order():
    scores = {"ADWIN": score(1.0, 50), "DDM": score(1.0, 20), "PH": score(1.0, 20)}
    assert rank_detectors(scores, ORDER) == ["DDM", "PH", "ADWIN"]


def test_silent_detector_ranks_last():
    scores = {
        "ADWIN": score(0.0, math.nan, alarms=0),
        "DDM": score(0.0, 900),
        "PH": score(0.1, 900),
    }
    assert rank_detectors(scores, ORDER) == ["PH", "DDM", "ADWIN"]


def test_write_best_detector_loads_as_config(tmp_path):
    trials = {}
    for name, f1, params in [
        ("ADWIN", 0.8, {"delta": 0.002}),
        ("PH", 0.9, {"delta": 0.005, "lambda_": 7}),
    ]:
        path = tmp_path / f"{name}.yml"
        OmegaConf.save(
            {
                "params": {"drift_detector": params},
                "user_attrs": {"dd_metrics": score(f1, 30)},
            },
            path,
        )
        trials[name] = path
    configs = {
        "ADWIN": "config/detector/ADWIN.yml",
        "PH": "config/detector/PH.yml",
    }
    target = tmp_path / "best.yml"
    write_best_detector(trials, configs, ["ADWIN", "PH"], "BEST", target)

    assert OmegaConf.load(target).selection.best == "PH"
    config = get_config(
        [
            "config/base.yml",
            "config/strategy/EWC.yml",
            "config/detector/BEST.yml",
            "config/boundary/abrupt.yml",
            target,
        ],
        [],
    )
    assert config.drift_detector.type_ == "PH"
    assert config.drift_detector.label == "BEST"
    assert config.drift_detector.lambda_ == 7
