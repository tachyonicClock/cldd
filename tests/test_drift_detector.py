from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.config import converter
from src.drift_detector import (
    ADWINArgs,
    DDMArgs,
    EDDMArgs,
    AnyDriftDetector,
    OracleArgs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DETECTOR_BASE_DIR = PROJECT_ROOT / "config" / "base" / "drift_detector"


def test_converter_rejects_unknown_drift_detector_type():
    with pytest.raises(Exception):
        converter.structure({"type_": "UNKNOWN"}, AnyDriftDetector)


def test_converter_rejects_extra_drift_detector_keys():
    with pytest.raises(Exception):
        converter.structure({"type_": "ADWIN", "unexpected": 1}, AnyDriftDetector)


def test_base_drift_detector_configs_parse_with_converter():
    config_paths = sorted(DRIFT_DETECTOR_BASE_DIR.glob("*.yml"))
    assert config_paths

    seen_types = set()

    for config_path in config_paths:
        config = OmegaConf.load(config_path)
        detector_config = config["drift_detector"]

        detector = converter.structure(detector_config, AnyDriftDetector)
        assert isinstance(detector, ADWINArgs | EDDMArgs | DDMArgs | OracleArgs)
        assert detector.type_ == detector_config["type_"]
        seen_types.add(detector.type_)

    assert seen_types == {"ADWIN", "EDDM", "DDM", "oracle"}
