from dataclasses import dataclass
from typing import ClassVar
from torch import nn
from capymoa.stream import Schema
from capymoa.ann import Perceptron, resnet20_32x32
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
class ResNet_32x32Args(ModelArgs):
    type_: ClassVar[str] = "resnet20_32x32"

    batch_norm: bool = True

    def build(self, seed: int, schema: Schema) -> nn.Module:
        manual_seed(seed)
        return resnet20_32x32(schema.get_num_classes(), self.batch_norm)


AnyModel = PerceptronArgs | ResNet_32x32Args
