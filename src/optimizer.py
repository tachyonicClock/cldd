from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterable

import torch
from torch.optim import AdamW

Params = Iterable[torch.nn.Parameter]


@dataclass
class OptimizerArgs(ABC):
    type_: ClassVar[str]

    lr: float = 0.001
    weight_decay: float = 0.0

    @abstractmethod
    def build_optimizer(self, params: Params) -> torch.optim.Optimizer: ...


@dataclass
class AdamArgs(OptimizerArgs):
    type_: ClassVar[str] = "Adam"

    def build_optimizer(self, params: Params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)


@dataclass
class AdamWArgs(OptimizerArgs):
    type_: ClassVar[str] = "AdamW"

    def build_optimizer(self, params: Params) -> torch.optim.Optimizer:
        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


AnyOptimizer = AdamWArgs | AdamArgs
