from dataclasses import dataclass
from typing import Optional, Sequence
from omegaconf import OmegaConf
from pathlib import Path
import cattrs
from cattrs.strategies import configure_tagged_union
from src.scenario import ScenarioArgs
from src.drift_detector import AnyDriftDetector
from src.learner import AnyLearner
from src.model import AnyModel
from src.hpsearch_config import HPSearchConfig, SuggestAny
import torch
import time


@dataclass
class Config:
    scenario: ScenarioArgs
    drift_detector: AnyDriftDetector
    learner: AnyLearner
    model: AnyModel

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mb_train: int = 64
    mb_test: int = 256

    seed: int = 0
    label: str = "noname"
    """Optional top level name for the experiment."""
    trial: Optional[int] = None
    hpsearch: Optional[HPSearchConfig] = None
    quiet: bool = False

    @property
    def scenario_label(self) -> str:
        return self.scenario.label

    @property
    def logdir(self) -> Path:
        if self.trial is None:
            trial_id = time.strftime("%Y%m%d-%H%M%S")
        else:
            trial_id = f"{self.trial:03d}"

        return (
            Path("logs")
            / self.label
            / self.scenario_label
            / self.learner.type_
            / (self.drift_detector.label or self.drift_detector.type_)
            / trial_id
        )

    @property
    def study_name(self) -> str:
        if self.hpsearch is None:
            raise ValueError("hpsearch config is required to generate study name.")
        return "/".join(
            [
                self.hpsearch.study_prefix,
                self.label,
                self.scenario_label,
                self.learner.type_,
                self.drift_detector.label or self.drift_detector.type_,
            ]
        )

    def apply_dotlist(self, dotlist: list[str]) -> "Config":
        return converter.structure(
            OmegaConf.merge(
                converter.unstructure(self),
                OmegaConf.from_dotlist(dotlist),
            ),
            Config,
        )

    def apply_dotlist_dict(self, dotlist_dict: dict) -> "Config":
        return self.apply_dotlist([f"{k}={v}" for k, v in dotlist_dict.items()])

    def dump(self) -> dict:
        return converter.unstructure(self)


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


def get_config(configs: Sequence[str | Path], dotlist: list[str]) -> Config:
    # Merge with precedence bases < config < dotlist
    bases = [OmegaConf.load(config) for config in configs]
    bases.append(OmegaConf.from_dotlist(dotlist))
    return converter.structure(OmegaConf.merge(*bases), Config)
