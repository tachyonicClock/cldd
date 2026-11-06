from capymoa.ocl import datasets
from capymoa.stream import Schema
from capymoa.ocl.datasets.gradual import gradual_sigmoid_transitions
from typing import Literal, Optional, Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import rotate
import numpy as np
import torch
from loguru import logger
from os import environ
from pathlib import Path

VALID_FRACTION = 0.1
ROTATED_TASKS = 5
CIFAR10C_BASE = 10_000
"""Images per severity in a CIFAR-10-C file: the CIFAR-10 test set."""
"""Tasks of RotatedCIFAR10, spread over 180 degrees like CapyMOA's rotated datasets."""
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
        "RotatedCIFAR10",
        "CIFAR10C",
        "Adult",
        "Credit",
        "Covertype",
    ]
    normalize_features: bool = True
    gradual: float = 0.0
    label: str = "unnamed-scenario"
    epochs: int = 1
    validation: bool = False
    image_size: int = 32
    """Image side in pixels, for datasets stored at several sizes (CLEAR10)."""
    rotation_range: float = 180.0
    """RotatedCIFAR10: domain i is rotated by rotation_range * i / 5 degrees."""
    grayscale: bool = False
    """RotatedCIFAR10: use luminance only, copied to 3 channels (removes colour cues)."""
    corruptions: Sequence[str] = (
        "clean", "gaussian_noise", "defocus_blur", "fog", "jpeg_compression"
    )
    """CIFAR10C: one domain per corruption, in this order (`clean` is plain CIFAR-10)."""
    cifar10c_test: Literal["official", "generated"] = "official"
    """CIFAR10C: test on the official CIFAR-10-C files or on our own corruptions."""
    cifar10c_disjoint: bool = False
    """CIFAR10C: give each domain its own fifth of the training images (10,000 each),
    fixed with `DOMAIN_SPLIT_SEED`, so no image is seen in two domains."""
    adult_partition: Literal["age", "occupation", "education"] = "age"
    """Adult: attribute that defines the domains (`data/adult[_occupation].pt`)."""
    credit_partition: Literal["SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL", "EDUCATION+LIMIT_BAL"] = "LIMIT_BAL"
    """Credit: attribute that defines the domains (`data/credit_<attr>.pt`)."""
    credit_rotated: bool = False
    """Credit: read the variant with per-domain feature rotation (`credit_<attr>_rot.pt`)."""
    debug_truncate: Optional[int] = None
    """If set, truncates each task to this many samples for faster debugging."""

    covertype_partition: Literal["Elevation", "Aspect", "Slope", "Horz_Dist_Roadways", "Horz_Dist_Hydrology", "Wilderness"] = "Elevation"
    """Covertype: attribute that defines the domains (`data/covertype_<attr>.pt`)."""

    def _get_clear10(self, seed: int) -> Scenario:
        """CLEAR10: 10 time buckets as domains, 11 classes (10 objects and background).

        Reads `data/clear10_<image_size>.pt`, made by `script/clear10.py`. The run seed
        shuffles instances within each bucket.
        """
        path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / f"clear10_{self.image_size}.pt"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `uv run script/clear10.py --size {self.image_size}`."
            )
        data = torch.load(path, map_location="cpu")
        mean = std = None
        if self.normalize_features:
            x = torch.cat([x for x, _ in data["train"]]).float() / 255.0
            mean = x.mean(dim=(0, 2, 3)).view(3, 1, 1)
            std = x.std(dim=(0, 2, 3)).view(3, 1, 1)
        train_tasks = [_ImageTask(x, y, mean, std) for x, y in data["train"]]
        test_tasks = [_ImageTask(x, y, mean, std) for x, y in data["test"]]
        n_classes = len(data["classes"])
        side = self.image_size
        schema = Schema.from_custom(
            features=[str(i) for i in range(3 * side * side)] + ["class"],
            target="class",
            categories={"class": [str(i) for i in range(n_classes)]},
        )
        schema.shape = (3, side, side)
        return Scenario(
            train_tasks=shuffle_task_contents(train_tasks, seed),
            test_tasks=test_tasks,
            schema=schema,
            mean=mean,
            std=std,
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

    def _get_rotated_cifar10(self, seed: int) -> Scenario:
        """CIFAR-10 with 5 domains, each a fixed rotation of the full set.

        Uses the angles of RotatedFashionMNIST (0, 36, 72, 108, 144 degrees) and the
        same nearest-neighbour rotation with black corners. Reads `data/cifar10.pt`,
        made by `script/cifar10.py`. The run seed shuffles instances within each task.
        """
        path = Path(__file__).resolve().parent.parent / "data" / "cifar10.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `uv run script/cifar10.py`."
            )
        data = torch.load(path, map_location="cpu")
        (x_train, y_train), (x_test, y_test) = data["train"], data["test"]
        if self.grayscale:
            x_train, x_test = _to_gray(x_train), _to_gray(x_test)
        mean = std = None
        if self.normalize_features:
            x = x_train.float() / 255.0
            mean = x.mean(dim=(0, 2, 3)).view(3, 1, 1)
            std = x.std(dim=(0, 2, 3)).view(3, 1, 1)
        angles = [self.rotation_range * i / ROTATED_TASKS for i in range(ROTATED_TASKS)]
        train_tasks = [_ImageTask(x_train, y_train, mean, std, a) for a in angles]
        test_tasks = [_ImageTask(x_test, y_test, mean, std, a) for a in angles]
        schema = Schema.from_custom(
            features=[str(i) for i in range(3 * 32 * 32)] + ["class"],
            target="class",
            categories={"class": [str(i) for i in range(10)]},
        )
        schema.shape = (3, 32, 32)
        return Scenario(
            train_tasks=shuffle_task_contents(train_tasks, seed),
            test_tasks=test_tasks,
            schema=schema,
            mean=mean,
            std=std,
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

    def _get_cifar10c(self, seed: int) -> Scenario:
        """CIFAR-10 under a sequence of corruptions, one domain per corruption.

        Training streams are the 50,000 CIFAR-10 training images corrupted with the
        CIFAR-10-C code at one random severity (1 to 5) each, from
        `data/cifar10c/<corruption>_train.npy` (`script/make_cifar10c.py`). Test sets are
        the official CIFAR-10-C images (Hendrycks and Dietterich, 2019) of the 10,000
        test images, one severity each, from `$DATASETS/CIFAR-10-C/<corruption>.npy`,
        or our own corruptions of them with `cifar10c_test="generated"`. The domain
        `clean` is plain CIFAR-10, and a suffix `_sK` (e.g. `fog_s5`) fixes severity K. The run seed shuffles instances within each domain.
        """
        root = Path(__file__).resolve().parent.parent / "data"
        clean = torch.load(root / "cifar10.pt", map_location="cpu")
        (x_clean_tr, y_tr), (x_clean_te, y_te) = clean["train"], clean["test"]
        official = Path(environ.get("DATASETS", "")) / "CIFAR-10-C"
        hwc_to_chw = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).contiguous()

        train, test = [], []
        for c in self.corruptions:
            if c == "clean":
                train.append((x_clean_tr, y_tr))
                test.append((x_clean_te, y_te))
                continue
            train.append((hwc_to_chw(np.load(root / "cifar10c" / f"{c}_train.npy")), y_tr))
            if self.cifar10c_test == "official":
                sev = np.load(root / "cifar10c" / f"{c}_test_severity.npy").astype(int)
                # `fog_s5` uses the official `fog.npy` at the severities stored for it
                arr = np.load(official / f"{c.split('_s')[0] if c[-3:-1] == '_s' else c}.npy", mmap_mode="r")
                x = arr[(sev - 1) * CIFAR10C_BASE + np.arange(CIFAR10C_BASE)]
            else:
                x = np.load(root / "cifar10c" / f"{c}_test.npy")
            test.append((hwc_to_chw(x), y_te))
        if self.cifar10c_disjoint:
            parts = np.array_split(np.random.default_rng(DOMAIN_SPLIT_SEED).permutation(len(y_tr)), len(train))
            train = [(x[idx], y[idx]) for (x, y), idx in zip(train, parts)]
        mean = std = None
        if self.normalize_features:
            x = torch.cat([x for x, _ in train]).float() / 255.0
            mean = x.mean(dim=(0, 2, 3)).view(3, 1, 1)
            std = x.std(dim=(0, 2, 3)).view(3, 1, 1)
        schema = Schema.from_custom(
            features=[str(i) for i in range(3 * 32 * 32)] + ["class"],
            target="class",
            categories={"class": [str(i) for i in range(10)]},
        )
        schema.shape = (3, 32, 32)
        return Scenario(
            train_tasks=shuffle_task_contents([_ImageTask(x, y, mean, std) for x, y in train], seed),
            test_tasks=[_ImageTask(x, y, mean, std) for x, y in test],
            schema=schema,
            mean=mean,
            std=std,
        )

    def _get_adult(self, seed: int) -> Scenario:
        """UCI Adult (income > 50K), one domain per age bin or occupation.

        Reads `data/adult[_occupation].pt`, made by `script/adult.py`.
        """
        suffix = "" if self.adult_partition == "age" else f"_{self.adult_partition}"
        return self._get_tabular(seed, f"adult{suffix}", f"script/adult.py --by {self.adult_partition}")

    def _get_credit(self, seed: int) -> Scenario:
        """UCI Default of Credit Card Clients, one domain per bin or value of `credit_partition`.

        Reads `data/credit_<credit_partition>.pt`, made by `script/credit.py`.
        """
        suffix = "_rot" if self.credit_rotated else ""
        return self._get_tabular(
            seed, f"credit_{self.credit_partition}{suffix}",
            f"script/credit.py --by {self.credit_partition}" + (" --rotate 180" if suffix else ""),
        )

    def _get_covertype(self, seed: int) -> Scenario:
        """UCI Covertype (7 cover types), one domain per bin of `covertype_partition`.

        Reads `data/covertype_<covertype_partition>.pt`, made by `script/covertype.py`.
        """
        return self._get_tabular(
            seed, f"covertype_{self.covertype_partition}",
            f"script/covertype.py --by {self.covertype_partition}",
        )

    def _get_tabular(self, seed: int, name: str, command: str) -> Scenario:
        """Tabular domains from `data/<name>.pt` (binary labels unless it stores `n_classes`)."""
        path = Path(__file__).resolve().parent.parent / "data" / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `uv run {command}`.")
        data = torch.load(path, map_location="cpu")
        n_features = data["train"][0][0].shape[1]
        schema = Schema.from_custom(
            features=[str(i) for i in range(n_features)] + ["class"],
            target="class",
            categories={"class": [str(i) for i in range(data.get("n_classes", 2))]},
        )
        return Scenario(
            train_tasks=shuffle_task_contents(
                [_RotatedFashionMNISTTask(x, y) for x, y in data["train"]], seed
            ),
            test_tasks=[_RotatedFashionMNISTTask(x, y) for x, y in data["test"]],
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
            case "RotatedCIFAR10":
                return self._get_rotated_cifar10(seed)
            case "CIFAR10C":
                return self._get_cifar10c(seed)
            case "Adult":
                return self._get_adult(seed)
            case "Credit":
                return self._get_credit(seed)
            case "Covertype":
                return self._get_covertype(seed)
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
    """uint8 images in memory, optionally rotated, then scaled to [0, 1] and normalized
    per channel."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
        angle: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.mean = mean
        self.std = std
        self.angle = angle

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self.x[index]
        if self.angle:
            # Same as CapyMOA's rotated datasets: nearest neighbour, black corners
            x = rotate(x, self.angle)
        x = x.float() / 255.0
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x, self.y[index]


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """uint8 (N, 3, H, W) to luminance (ITU-R BT.601), copied to 3 channels."""
    w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    g = (x.float() * w).sum(dim=1, keepdim=True).round().clamp(0, 255).to(torch.uint8)
    return g.expand(-1, 3, -1, -1).contiguous()
