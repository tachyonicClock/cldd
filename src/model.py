from dataclasses import dataclass
from typing import ClassVar
from torch import nn
from capymoa.stream import Schema
from capymoa.ann import Perceptron
from torch import manual_seed


@dataclass
class ModelArgs:
    type_: ClassVar[str]

    def build(self, seed: int, schema: Schema) -> nn.Module:
        raise NotImplementedError("Must implement build method for model config")


@dataclass
class PerceptronArgs(ModelArgs):
    type_: ClassVar[str] = "MLP"

    def build(self, seed: int, schema: Schema) -> nn.Module:
        manual_seed(seed)
        return Perceptron(schema)


@dataclass
class ResNet18Args(ModelArgs):
    type_: ClassVar[str] = "ResNet18"


AnyModel = PerceptronArgs | ResNet18Args
