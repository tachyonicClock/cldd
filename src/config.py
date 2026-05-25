from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import cattrs
from cattrs.strategies import configure_tagged_union
from src.scenario import ScenarioArgs
from src.drift_detector import AnyDriftDetector
from src.learner import AnyLearner
from src.model import AnyModel


@dataclass
class Config:
    name: str
    scenario: ScenarioArgs
    drift_detector: AnyDriftDetector
    learner: AnyLearner
    model: AnyModel

    device: str = "cpu"
    mb_train: int = 64
    mb_eval: int = 256

    seed: int = 0
    bases: Optional[list[str]] = None


# Setup cattrs converter for auto-disambiguation of union types.
converter = cattrs.Converter()
converter.forbid_extra_keys = True
configure_tagged_union(
    AnyDriftDetector, converter, tag_name="type_", tag_generator=lambda cls: cls.type_
)
configure_tagged_union(
    AnyLearner, converter, tag_name="type_", tag_generator=lambda cls: cls.type_
)
configure_tagged_union(
    AnyModel, converter, tag_name="type_", tag_generator=lambda cls: cls.type_
)


def get_config(config_path: Path, bases_dir: Path, dotlist: list[str]) -> Config:
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a DictConfig, got {type(config)}")

    # Merge with precedence bases < config < dotlist
    bases = [OmegaConf.load(bases_dir / base) for base in config.get("bases", [])]
    bases.append(config)
    bases.append(OmegaConf.from_dotlist(dotlist))

    return converter.structure(OmegaConf.merge(*bases), Config)
