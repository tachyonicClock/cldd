from capymoa.ocl import datasets
from capymoa.stream import Schema
from capymoa.stream.torch import TorchStream
from capymoa.ocl.datasets.gradual import gradual_sigmoid_transitions
from typing import Literal, Optional, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import torch
from loguru import logger
from os import environ
from pathlib import Path
from src.dataset.clear10 import load_clear10

VALID_FRACTION = 0.1
DOMAIN_SPLIT_SEED = 0
"""Seed that picks which fine classes form each DomainCIFAR100 domain."""


def split_train_valid(train_tasks: Sequence[Dataset], valid_fraction: float, seed: int):
    if valid_fraction <= 0.0:
        return train_tasks, None
    elif valid_fraction >= 1.0:
        return None, train_tasks

    rng = np.random.default_rng(seed)
    valid_tasks = []
    new_train_tasks = []
    for task in train_tasks:
        n_samples = len(task)  # type: ignore
        n_valid = int(n_samples * valid_fraction)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        valid_indices = indices[:n_valid]
        train_indices = indices[n_valid:]

        valid_tasks.append(Subset(task, valid_indices))  # type: ignore
        new_train_tasks.append(Subset(task, train_indices))  # type: ignore

    return new_train_tasks, valid_tasks


def shuffle_task_contents(tasks: Sequence[Dataset], seed: int) -> Sequence[Dataset]:
    rng = np.random.default_rng(seed)
    shuffled_tasks = []
    for task in tasks:
        indices = np.arange(len(task))  # type: ignore
        rng.shuffle(indices)
        shuffled_tasks.append(Subset(task, indices))  # type: ignore
    return shuffled_tasks


@dataclass
class Scenario:
    train_tasks: Sequence[Dataset]
    test_tasks: Sequence[Dataset]
    schema: Schema
    mean: Optional[torch.Tensor] = None
    """Per-channel mean used to normalize images, if known."""
    std: Optional[torch.Tensor] = None


@dataclass
class ScenarioArgs:
    name: Literal[
        "DomainCIFAR100ViT",
        "DomainCIFAR100",
        "RotatedMNIST",
        "RotatedFashionMNIST",
        "RotatedFashionMNISTPT",
        "RotatedTinyMNIST",
        "CLEAR10",
        "CORe50",
    ]
    normalize_features: bool = True
    gradual: float = 0.0
    label: str = "unnamed-scenario"
    epochs: int = 1
    validation: bool = False
    debug_truncate: Optional[int] = None
    """If set, truncates each task to this many samples for faster debugging."""

    def _get_clear10(self, seed: int) -> Scenario:
        root = environ["DATASETS"]
        if self.normalize_features:
            transform = Compose(
                [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )
        else:
            transform = Compose([ToTensor()])

        train_tasks = load_clear10(Path(root), "train", transform)
        test_tasks = load_clear10(Path(root), "test", transform)
        train_tasks = shuffle_task_contents(train_tasks, seed)

        stream = TorchStream.from_classification(
            train_tasks[0],
            num_classes=10,
            dataset_name="CLEAR10",
            shape=[3, 224, 224],
            shuffle=False,
        )

        return Scenario(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            schema=stream.get_schema(),
        )

    def _get_domain_cifar100_vit(self, seed: int) -> Scenario:
        # Fix the fine classes in each domain, so every seed sees the same
        # distributions in the same order (as in RotatedFashionMNISTPT). The run
        # seed only shuffles instances within each domain. ViT features are
        # already normalized.
        dataset = datasets.DomainCIFAR100ViT(seed=DOMAIN_SPLIT_SEED)
        return Scenario(
            train_tasks=shuffle_task_contents(dataset.train_tasks, seed),
            test_tasks=dataset.test_tasks,
            schema=dataset.schema,
        )

    def _get_domain_cifar100(self, seed: int) -> Scenario:
        # Same fixed domain split as DomainCIFAR100ViT, on raw pixels.
        dataset = datasets.DomainCIFAR100(
            seed=DOMAIN_SPLIT_SEED, normalize_features=self.normalize_features
        )
        return Scenario(
            train_tasks=shuffle_task_contents(dataset.train_tasks, seed),
            test_tasks=dataset.test_tasks,
            schema=dataset.schema,
        )

    def _get_core50(self, seed: int) -> Scenario:
        """CORe50 with each of its 11 sessions as a domain (new instances, NI).

        Sessions change background and lighting. The label is one of 50 objects.
        Reads `$DATASETS/core50_32x32.hdf5`. The run seed shuffles instances within
        each session.
        """
        import h5py

        path = Path(environ["DATASETS"]) / "core50_32x32.hdf5"
        with h5py.File(path, "r") as f:
            sessions = sorted(f.keys(), key=lambda k: int(k[1:]))
            train = [
                (
                    torch.from_numpy(f[f"{s}/train/features"][:]),
                    torch.from_numpy(f[f"{s}/train/labels"][:]).long(),
                )
                for s in sessions
            ]
            test = [
                (
                    torch.from_numpy(f[f"{s}/test/features"][:]),
                    torch.from_numpy(f[f"{s}/test/labels"][:]).long(),
                )
                for s in sessions
            ]

        mean = std = None
        if self.normalize_features:
            x = torch.cat([x for x, _ in train]).float() / 255.0
            mean = x.mean(dim=(0, 2, 3)).view(3, 1, 1)
            std = x.std(dim=(0, 2, 3)).view(3, 1, 1)

        train_tasks = [_ImageTask(x, y, mean, std) for x, y in train]
        test_tasks = [_ImageTask(x, y, mean, std) for x, y in test]
        schema = Schema.from_custom(
            features=[str(i) for i in range(3 * 32 * 32)] + ["class"],
            target="class",
            categories={"class": [str(i) for i in range(50)]},
        )
        schema.shape = (3, 32, 32)
        return Scenario(
            train_tasks=shuffle_task_contents(train_tasks, seed),
            test_tasks=test_tasks,
            schema=schema,
            mean=mean,
            std=std,
        )

    def _get_rotated_fashion_mnist_pt(self) -> Scenario:
        data_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "rotated_fashion_mnist_data.pt"
        )
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at {data_path}. Expected data/rotated_fashion_mnist_data.pt"
            )

        dataset = torch.load(data_path, map_location="cpu")

        train_tasks = [_RotatedFashionMNISTTask(x, y) for x, y in dataset["train"]]
        test_tasks = [_RotatedFashionMNISTTask(x, y) for x, y in dataset["test"]]

        schema = Schema.from_custom(
            features=[str(i) for i in range(28 * 28)] + ["class"],
            target="class",
            categories={"class": [str(i) for i in range(10)]},
        )
        schema.shape = (1, 28, 28)

        return Scenario(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            schema=schema,
        )

    def _get_dataset(self, seed: int):
        kwargs = dict(
            seed=seed,
            normalize_features=self.normalize_features,
        )
        match self.name:
            case "DomainCIFAR100ViT":
                return self._get_domain_cifar100_vit(seed)
            case "DomainCIFAR100":
                return self._get_domain_cifar100(seed)
            case "RotatedMNIST":
                return datasets.RotatedMNIST(**kwargs, preload_test=False)
            case "RotatedFashionMNIST":
                return datasets.RotatedFashionMNIST(**kwargs, preload_test=False)
            case "RotatedFashionMNISTPT":
                return self._get_rotated_fashion_mnist_pt()
            case "RotatedTinyMNIST":
                return datasets.RotatedTinyMNIST(**kwargs, preload_test=False)
            case "CLEAR10":
                return self._get_clear10(seed)
            case "CORe50":
                return self._get_core50(seed)
            case _:
                raise ValueError(f"Unknown scenario name: {self.name}")

    def build(self, seed: int) -> Scenario:
        scenario = self._get_dataset(seed)
        train_tasks = scenario.train_tasks
        test_tasks = scenario.test_tasks

        if self.debug_truncate is not None:
            logger.warning(
                "`scenario.debug_truncate` is set. Truncating each task to {} samples.",
                self.debug_truncate,
            )
            train_tasks = [
                Subset(task, list(range(min(len(task), self.debug_truncate))))  # type: ignore
                for task in train_tasks
            ]
            test_tasks = [
                Subset(task, list(range(min(len(task), self.debug_truncate))))  # type: ignore
                for task in test_tasks
            ]

        if self.validation:
            logger.info(
                "Splitting train tasks into train and valid sets with fraction {}.",
                VALID_FRACTION,
            )
            train_tasks, test_tasks = split_train_valid(
                train_tasks, VALID_FRACTION, seed
            )

        if self.gradual > 0.0:
            train_tasks = gradual_sigmoid_transitions(train_tasks, width=self.gradual)

        return Scenario(
            train_tasks=train_tasks,  # type: ignore
            test_tasks=test_tasks,  # type: ignore
            schema=scenario.schema,
            mean=getattr(scenario, "mean", None),
            std=getattr(scenario, "std", None),
        )


class _RotatedFashionMNISTTask(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self.x[index]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return x, self.y[index]


class _ImageTask(Dataset):
    """uint8 images in memory, scaled to [0, 1] and normalized per channel."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
    ):
        self.x = x
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self.x[index].float() / 255.0
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x, self.y[index]
