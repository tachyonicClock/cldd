from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.config import converter
from src.learner import (
    DERArgs,
    ERArgs,
    EWCArgs,
    FTArgs,
    LWFArgs,
    PNArgs,
    RARArgs,
    SIArgs,
    AnyLearner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEARNER_BASE_DIR = PROJECT_ROOT / "config" / "base" / "learner"


def test_converter_rejects_unknown_learner_type():
    with pytest.raises(Exception):
        converter.structure({"type_": "UNKNOWN"}, AnyLearner)


def test_converter_rejects_extra_learner_keys():
    with pytest.raises(Exception):
        converter.structure({"type_": "EWC", "unexpected": 1}, AnyLearner)


def test_base_learner_configs_parse_with_converter():
    config_paths = sorted(LEARNER_BASE_DIR.glob("*.yml"))
    assert config_paths

    seen_types = set()

    for config_path in config_paths:
        config = OmegaConf.load(config_path)
        learner_config = config["learner"]

        learner = converter.structure(learner_config, AnyLearner)
        assert isinstance(
            learner,
            EWCArgs | SIArgs | LWFArgs | DERArgs | PNArgs | RARArgs | ERArgs | FTArgs,
        )
        assert learner.type_ == learner_config["type_"]
        seen_types.add(learner.type_)

    assert seen_types == {"EWC", "SI", "LWF", "DER", "PN", "RAR", "ER", "FT"}
