from dataclasses import dataclass
from typing import ClassVar
from torch import nn


@dataclass
class ModelArgs:
    type_: ClassVar[str]

    def build(self, seed: int, n_classes: int) -> nn.Module:
        raise NotImplementedError("Must implement build method for model config")


@dataclass
class MLPArgs(ModelArgs):
    type_: ClassVar[str] = "MLP"


@dataclass
class ResNet18Args(ModelArgs):
    type_: ClassVar[str] = "ResNet18"


AnyModel = MLPArgs | ResNet18Args
