from dataclasses import dataclass
from capymoa.stream import Schema
from typing import ClassVar, Literal, override
from capymoa.base import BatchClassifier
from torch import nn
from capymoa.classifier import Finetune
from capymoa.ocl.strategy import EWC, ExperienceReplay, LWF, DER
from torch.optim import Adam
from abc import ABC, abstractmethod
import torchvision.transforms as T
import torch


AugmentTypes = Literal["Dropout", "AutoAugCIFAR10"]


class _AutoAugmentCIFAR10(nn.Module):
    mean = [0.507, 0.487, 0.441]
    std = [0.267, 0.256, 0.276]

    def __init__(self):
        super().__init__()
        self._augment = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
        mean = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            return self._mean, self._std
        if x.dim() == 3:
            return self._mean[0], self._std[0]
        raise ValueError(f"Expected a 3D or 4D tensor, got shape={tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3] != 3:
            raise ValueError(
                f"AutoAugCIFAR10 expects 3 channels, got shape={tuple(x.shape)}"
            )

        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        mean, std = self._stats(x_float)

        # Unnormalize -> uint8 -> AutoAugment -> normalize back.
        x_img = x_float * std + mean
        x_uint8 = (x_img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        x_uint8 = self._augment(x_uint8)
        x_img = x_uint8.to(torch.float32) / 255.0
        x_norm = (x_img - mean) / std

        if torch.is_floating_point(x):
            return x_norm.to(input_dtype)
        return x_norm


def build_augment(augment_type: AugmentTypes) -> nn.Module:
    if augment_type == "Dropout":
        return nn.Dropout(p=0.5)
    elif augment_type == "AutoAugCIFAR10":
        return _AutoAugmentCIFAR10()
    else:
        raise ValueError(f"Invalid augment_type: {augment_type}")


@dataclass
class LearnerArgs(ABC):
    type_: ClassVar[str]
    lr: float = 0.001
    """Learning rate for the optimizer used to train the model."""

    @abstractmethod
    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier: ...


@dataclass
class FTArgs(LearnerArgs):
    type_: ClassVar[str] = "FT"

    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
        return Finetune(
            schema,
            model,
            optimizer=Adam(model.parameters(), lr=self.lr),
            device=device,
            random_seed=seed,
        )


@dataclass
class EWCArgs(LearnerArgs):
    type_: ClassVar[str] = "EWC"

    lambda_: float = 1000.0
    """Weight of the EWC regularisation term."""
    gamma: float = 1.0
    """Discount factor for the importance of previous tasks. Should be in [0, 1].
    Usually close to 1.0."""
    fim_batch_size: int = 32
    """Batch size for computing the Fisher Information Matrix (FIM)."""
    buffer_capacity: int = 256
    """Capacity of the buffer used to store samples for FIM estimation."""

    @override
    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
        return EWC(
            schema=schema,
            model=model,
            optimiser=Adam(model.parameters(), lr=self.lr),
            lambda_=self.lambda_,
            buffer_capacity=self.buffer_capacity,
            fim_batch_size=self.fim_batch_size,
            device=torch.device(device),
            gamma=self.gamma,
        )


# @dataclass
# class SIArgs(LearnerArgs):
#     type_: ClassVar[str] = "SI"


@dataclass
class LWFArgs(LearnerArgs):
    type_: ClassVar[str] = "LWF"
    alpha: float = 1.0
    """Weight of the distillation loss term."""
    temperature: float = 2.0
    """Distillation temperature."""

    @override
    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
        model.to(device)
        return LWF(
            schema=schema,
            model=model,
            optimiser=Adam(model.parameters(), lr=self.lr),
            device=torch.device(device),
            alpha=self.alpha,
            temperature=self.temperature,
        )


@dataclass
class DERArgs(LearnerArgs):
    type_: ClassVar[str] = "DER"
    buffer_capacity: int = 256
    """Capacity of the experience replay buffer."""
    alpha: float = 0.5
    """Weight of the DER replay-logit loss term."""

    augment: AugmentTypes = "AutoAugCIFAR10"

    @override
    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
        model.to(device)

        return DER(
            schema=schema,
            model=model,
            optimiser=Adam(model.parameters(), lr=self.lr),
            device=torch.device(device),
            alpha=self.alpha,
            buffer_capacity=self.buffer_capacity,
            seed=seed,
            augment=build_augment(self.augment),
        )


# @dataclass
# class PNArgs(LearnerArgs):
#     type_: ClassVar[str] = "PN"


# @dataclass
# class RARArgs(LearnerArgs):
#     type_: ClassVar[str] = "RAR"


@dataclass
class ERArgs(LearnerArgs):
    type_: ClassVar[str] = "ER"
    buffer_capacity: int = 256
    """Capacity of the experience replay buffer."""

    @override
    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
        return ExperienceReplay(
            learner=Finetune(
                schema,
                model,
                optimizer=Adam(model.parameters(), lr=self.lr),
                device=device,
                random_seed=seed,
            ),
            buffer_capacity=self.buffer_capacity,
        )


AnyLearner = FTArgs | EWCArgs | ERArgs | LWFArgs | DERArgs
