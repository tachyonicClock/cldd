from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal, cast, override

import torch
import torchvision.transforms as T
from capymoa.base import BatchClassifier
from capymoa.classifier import Finetune
from capymoa.ocl.datasets import DomainCIFAR100
from capymoa.ocl.evaluation.events import TrainTaskBegin
from capymoa.ocl.strategy import EWC, LWF, MAS, SI, RWalk
from capymoa.ocl.util._optim import reset_optimizer_state
from capymoa.stream import Schema
from torch import nn
from torch.optim import Optimizer

AugmentTypes = Literal["Dropout", "AutoAugCIFAR10"]

DEFAULT_BUFFER_CAPACITY = 1_000
AUX_BATCH_SIZE = 64
"""Batch size for auxiliary computations such as FIM estimation in EWC."""
SUBSTEPS = 1


class _AutoAugmentCIFAR10(nn.Module):
    mean = DomainCIFAR100.mean
    std = DomainCIFAR100.std

    def __init__(self):
        super().__init__()
        self._augment = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
        mean = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = cast(torch.Tensor, self._mean)
        std = cast(torch.Tensor, self._std)
        if x.dim() == 4:
            return mean, std
        if x.dim() == 3:
            return mean[0], std[0]
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

    @abstractmethod
    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier: ...


@dataclass
class FTArgs(LearnerArgs):
    """Finetuning"""

    type_: ClassVar[str] = "FT"

    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier:
        return Finetune(
            schema,
            model,
            optimizer=optimizer,
            device=device,
            random_seed=seed,
        )


class _EWC(EWC):
    """EWC that resets the optimizer state on each new task.

    CapyMOA 0.15.1 dropped this reset from EWC, but the other strategies and the
    version used for earlier runs still do it.
    """

    def on_train_task(self, event: TrainTaskBegin) -> None:
        reset_optimizer_state(self._optimiser)
        super().on_train_task(event)


@dataclass
class EWCArgs(LearnerArgs):
    """Elastic Weight Consolidation"""

    type_: ClassVar[str] = "EWC"

    lambda_: float = 1000.0
    """Weight of the EWC regularisation term."""
    gamma: float = 1.0
    """Discount factor for the importance of previous tasks. Should be in [0, 1].
    Usually close to 1.0."""
    buffer_capacity: int = DEFAULT_BUFFER_CAPACITY
    """Capacity of the buffer used to store samples for FIM estimation."""

    @override
    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier:
        return _EWC(
            schema=schema,
            model=model,
            optimiser=optimizer,
            lambda_=self.lambda_,
            fim_buffer=self.buffer_capacity,
            fim_batch_size=AUX_BATCH_SIZE,
            device=torch.device(device),
            gamma=self.gamma,
        )


@dataclass
class SIArgs(LearnerArgs):
    """Synaptic Intelligence"""

    type_: ClassVar[str] = "SI"

    lambda_: float = 1.0
    """Weight of the SI regularisation term."""
    damping: float = 0.1
    """Damping value used during SI importance consolidation."""

    @override
    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier:
        return SI(
            schema=schema,
            model=model,
            optimiser=optimizer,
            lambda_=self.lambda_,
            damping=self.damping,
            device=torch.device(device),
        )


@dataclass
class LWFArgs(LearnerArgs):
    """Learning Without Forgetting"""

    type_: ClassVar[str] = "LWF"
    alpha: float = 1.0
    """Weight of the distillation loss term."""
    temperature: float = 2.0
    """Distillation temperature."""

    @override
    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier:
        model.to(device)
        return LWF(
            schema=schema,
            model=model,
            optimiser=optimizer,
            device=torch.device(device),
            alpha=self.alpha,
            temperature=self.temperature,
        )


@dataclass
class RWalkArgs(LearnerArgs):
    """Riemannian Walk"""

    type_: ClassVar[str] = "RWalk"

    lambda_: float = 1.0
    """Weight of the RWalk regularisation term."""
    alpha: float = 0.9
    """Exponential moving average factor for importance scores."""
    delta_t: int = 10
    """Number of steps between importance score updates."""

    @override
    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier:
        return RWalk(
            schema=schema,
            model=model,
            optimiser=optimizer,
            lambda_=self.lambda_,
            alpha=self.alpha,
            delta_t=self.delta_t,
            device=torch.device(device),
        )


@dataclass
class MASArgs(LearnerArgs):
    """Memory Aware Synapses"""

    type_: ClassVar[str] = "MAS"

    lambda_: float = 1.0
    """Weight of the MAS regularisation term."""
    alpha: float = 0.5
    """Exponential moving average factor for importance scores."""
    buffer_capacity: int = DEFAULT_BUFFER_CAPACITY
    """Capacity of the buffer used to estimate importance scores."""

    @override
    def build(
        self,
        seed: int,
        schema: Schema,
        device: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> BatchClassifier:
        return MAS(
            schema=schema,
            model=model,
            optimiser=optimizer,
            lambda_=self.lambda_,
            alpha=self.alpha,
            buffer_capacity=self.buffer_capacity,
            importance_batch_size=AUX_BATCH_SIZE,
            device=torch.device(device),
        )


AnyLearner = FTArgs | EWCArgs | LWFArgs | SIArgs | RWalkArgs | MASArgs
