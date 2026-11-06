import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cattrs
import torch
from cattrs.strategies import configure_tagged_union
from omegaconf import OmegaConf

from src.drift_detector import AnyDriftDetector
from src.hpsearch_config import HPSearchConfig, SuggestAny
from src.model import AnyModel
from src.optimizer import AnyOptimizer
from src.scenario import ScenarioArgs
from src.scheduler import AnyScheduler
from src.strategy import AnyLearner

DEFAULT_DATASET_LABEL = "rfmnist"
"""Dataset name used in results when `Config.dataset_label` is None."""


@dataclass
class Config:
    scenario: ScenarioArgs
    drift_detector: AnyDriftDetector
    learner: AnyLearner
    model: AnyModel
    optimizer: AnyOptimizer
    scheduler: AnyScheduler | None = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mb_train: int = 64
    train_repeats: int = 1
    """Training steps on each mini-batch. Values above 1 help single-pass convergence."""
    augment_repeats: bool = False
    """Augment the batch on the extra steps."""
    repeat_augment: Literal["autoaugment", "flipcrop"] = "autoaugment"
    """AutoAugment (CIFAR-10 policy, includes rotations; needs scenario mean/std) or
    random flip and crop (no rotation)."""
    mb_test: int = 256
    disable_progress_bar: bool = False

    seed: int = 0
    label: str = "noname"
    """Optional top level name for the experiment."""
    dataset_label: str | None = None
    """Adds a dataset level to log paths and study names. None keeps the
    RotatedFashionMNIST layout."""
    trial: int | None = None
    hpsearch: HPSearchConfig | None = None

    include: Sequence[Path] | None = None
    """List of config files to include. NOT RECURSIVE."""

    @property
    def dataset_name(self) -> str:
        return self.dataset_label or DEFAULT_DATASET_LABEL

    @property
    def scenario_label(self) -> str:
        return self.scenario.label

    @property
    def logdir(self) -> Path:
        if self.trial is None:
            trial_id = time.strftime("%Y%m%d-%H%M%S")
        else:
            trial_id = f"{self.trial:03d}"

        root = Path(os.environ.get("LOGDIR", "logs")) / "cldd"
        if self.dataset_label is not None:
            root = root / self.dataset_label
        return (
            root
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
                *([self.dataset_label] if self.dataset_label else []),
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
# converter.forbid_extra_keys = True


def type_tagged_union(type_):
    return configure_tagged_union(
        type_, converter, tag_name="type_", tag_generator=lambda cls: cls.type_
    )


type_tagged_union(AnyDriftDetector)
type_tagged_union(AnyLearner)
type_tagged_union(AnyModel)
type_tagged_union(AnyOptimizer)  # TODO: Re-enable when we have more optimizers
# type_tagged_union(AnyScheduler) # TODO: Re-enable when we have more schedulers
type_tagged_union(SuggestAny)


def get_config(configs: Sequence[str | Path], dotlist: list[str]) -> Config:

    configs_merged = OmegaConf.merge(*(OmegaConf.load(c) for c in configs))

    include = list(configs_merged.get("include", []))  # type: ignore
    if len(include) > 0:
        include_merged = OmegaConf.merge(*(OmegaConf.load(i) for i in include))
    else:
        include_merged = {}

    dotlist_dict = OmegaConf.from_dotlist(dotlist)

    # Merge with precedence includes < configs < dotlist
    all_merged = OmegaConf.merge(include_merged, configs_merged, dotlist_dict)
    return converter.structure(all_merged, Config)
