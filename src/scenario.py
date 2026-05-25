from capymoa.ocl import datasets
from capymoa.stream import Schema
from capymoa.ocl.datasets.fuzzy import fuzzy_sigmoid_transitions
from typing import Literal, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class Scenario:
    train_tasks: Sequence[Dataset]
    test_tasks: Sequence[Dataset]
    schema: Schema
    valid_tasks: Sequence[Dataset] | None = None


@dataclass
class ScenarioArgs:
    name: Literal[
        "DomainCIFAR100ViT",
        "DomainCIFAR100",
        "RotatedMNIST",
        "RotatedFashionMNIST",
    ]
    normalize_features: bool = True

    gradual: bool = False
    gradual_width: float = 0.5

    def _get_dataset(self, seed: int):
        match self.name:
            case "DomainCIFAR100ViT":
                assert (
                    not self.normalize_features
                )  # ViT features are already normalized
                return datasets.DomainCIFAR100ViT(seed=seed)
            case "DomainCIFAR100":
                return datasets.DomainCIFAR100(
                    seed=seed, normalize_features=self.normalize_features
                )
            case "RotatedMNIST":
                return datasets.RotatedMNIST(
                    seed=seed, normalize_features=self.normalize_features
                )
            case "RotatedFashionMNIST":
                return datasets.RotatedFashionMNIST(
                    seed=seed, normalize_features=self.normalize_features
                )
            case _:
                raise ValueError(f"Unknown scenario name: {self.name}")

    def build(self, seed: int) -> Scenario:
        scenario = self._get_dataset(seed)
        train_tasks = scenario.train_tasks

        if self.gradual:
            train_tasks = fuzzy_sigmoid_transitions(
                train_tasks, width=self.gradual_width
            )

        return Scenario(
            train_tasks=train_tasks,
            test_tasks=scenario.test_tasks,
            schema=scenario.schema,
            valid_tasks=None,
        )
