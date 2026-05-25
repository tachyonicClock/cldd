from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.config import converter
from src.model import AnyModel, MLPArgs, ResNet18Args


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BASE_DIR = PROJECT_ROOT / "config" / "base" / "model"


def test_converter_rejects_unknown_model_type():
    with pytest.raises(Exception):
        converter.structure({"type_": "UNKNOWN"}, AnyModel)


def test_converter_rejects_extra_model_keys():
    with pytest.raises(Exception):
        converter.structure({"type_": "MLP", "unexpected": 1}, AnyModel)


def test_base_model_configs_parse_with_converter():
    config_paths = sorted(MODEL_BASE_DIR.glob("*.yml"))
    assert config_paths

    seen_types = set()

    for config_path in config_paths:
        config = OmegaConf.load(config_path)
        model_config = config["model"]

        model = converter.structure(model_config, AnyModel)
        assert isinstance(model, MLPArgs | ResNet18Args)
        assert model.type_ == model_config["type_"]
        seen_types.add(model.type_)

    assert seen_types == {"MLP", "ResNet18"}
