from capymoa.ocl import datasets
from typing import Literal
from dataclasses import dataclass


@dataclass
class Scenario:
    name: Literal[
        "DomainCIFAR100ViT",
        "DomainCIFAR100",
        "RotatedMNIST",
        "RotatedFashionMNIST",
    ]
    seed: int
    normalize_features: bool = True

    gradual: bool = False
    gradual_width: float = 0.5

    def _get_dataset(self) -> datasets.Scenario:
        match self.name:
            case "DomainCIFAR100ViT":
                assert (
                    not self.normalize_features
                )  # ViT features are already normalized
                return datasets.DomainCIFAR100ViT(seed=self.seed)
            case "DomainCIFAR100":
                return datasets.DomainCIFAR100(
                    seed=self.seed, normalize_features=self.normalize_features
                )
            case "RotatedMNIST":
                return datasets.RotatedMNIST(
                    seed=self.seed, normalize_features=self.normalize_features
                )
            case "RotatedFashionMNIST":
                return datasets.RotatedFashionMNIST(
                    seed=self.seed, normalize_features=self.normalize_features
                )
            case _:
                raise ValueError(f"Unknown scenario name: {self.name}")

    def build(self) -> datasets.Scenario:
        scenario = self._get_dataset()
        scenario.train_tasks = fuzzy()
