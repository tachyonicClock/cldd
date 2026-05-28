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
        "RotatedTinyMNIST",
    ]
    normalize_features: bool = True
    gradual: float = 0.0
    label: str = "unnamed-scenario"

    def _get_dataset(self, seed: int):
        kwargs = dict(
            seed=seed,
            normalize_features=self.normalize_features,
        )
        match self.name:
            case "DomainCIFAR100ViT":
                # ViT features are already normalized
                assert not self.normalize_features
                return datasets.DomainCIFAR100ViT(**kwargs)
            case "DomainCIFAR100":
                return datasets.DomainCIFAR100(**kwargs)
            case "RotatedMNIST":
                return datasets.RotatedMNIST(**kwargs, preload_test=False)
            case "RotatedFashionMNIST":
                return datasets.RotatedFashionMNIST(**kwargs, preload_test=False)
            case "RotatedTinyMNIST":
                return datasets.RotatedTinyMNIST(**kwargs, preload_test=False)
            case _:
                raise ValueError(f"Unknown scenario name: {self.name}")

    def build(self, seed: int) -> Scenario:
        scenario = self._get_dataset(seed)
        train_tasks = scenario.train_tasks

        if self.gradual > 0.0:
            train_tasks = fuzzy_sigmoid_transitions(train_tasks, width=self.gradual)

        return Scenario(
            train_tasks=train_tasks,
            test_tasks=scenario.test_tasks,
            schema=scenario.schema,
            valid_tasks=None,
        )
