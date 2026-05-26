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
from src.hpsearch_config import HPSearchConfig, SuggestAny
import torch


@dataclass
class Config:
    scenario: ScenarioArgs
    drift_detector: AnyDriftDetector
    learner: AnyLearner
    model: AnyModel

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mb_train: int = 64
    mb_eval: int = 256

    seed: int = 0
    label: str = "noname"
    """Optional top level name for the experiment."""
    trial: Optional[int] = None
    bases: Optional[list[str]] = None
    hpsearch: Optional[HPSearchConfig] = None

    @property
    def scenario_label(self) -> str:
        return f"{self.scenario.name}_{self.scenario.gradual}"

    @property
    def method_label(self) -> str:
        return f"{self.learner.type_}_{self.drift_detector.type_}_{self.model.type_}"


# Setup cattrs converter for auto-disambiguation of union types.
converter = cattrs.Converter()
converter.forbid_extra_keys = True


def type_tagged_union(type_):
    return configure_tagged_union(
        type_, converter, tag_name="type_", tag_generator=lambda cls: cls.type_
    )


type_tagged_union(AnyDriftDetector)
type_tagged_union(AnyLearner)
type_tagged_union(AnyModel)
type_tagged_union(SuggestAny)


def get_config(config_path: Path, bases_dir: Path, dotlist: list[str]) -> Config:
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a DictConfig, got {type(config)}")

    # Merge with precedence bases < config < dotlist
    bases = [OmegaConf.load(bases_dir / base) for base in config.get("bases", [])]
    bases.append(config)
    bases.append(OmegaConf.from_dotlist(dotlist))

    return converter.structure(OmegaConf.merge(*bases), Config)
