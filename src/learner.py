from dataclasses import dataclass
from capymoa.stream import Schema
from typing import ClassVar, override
from capymoa.base import BatchClassifier
from torch import nn
from capymoa.classifier import Finetune
from capymoa.ocl.strategy import EWC
from torch.optim import Adam
from abc import ABC, abstractmethod
import torch


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


# @dataclass
# class LWFArgs(LearnerArgs):
#     type_: ClassVar[str] = "LWF"


# @dataclass
# class DERArgs(LearnerArgs):
#     type_: ClassVar[str] = "DER"


# @dataclass
# class PNArgs(LearnerArgs):
#     type_: ClassVar[str] = "PN"


# @dataclass
# class RARArgs(LearnerArgs):
#     type_: ClassVar[str] = "RAR"


# @dataclass
# class ERArgs(LearnerArgs):
#     type_: ClassVar[str] = "ER"




AnyLearner = FTArgs | EWCArgs
