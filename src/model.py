from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from torch import nn
from capymoa.stream import Schema
from capymoa.ann import Perceptron, resnet20_32x32
from torch import manual_seed
import torch

file = Path(__file__).resolve()

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
    pretrained: bool = True

    def build(self, seed: int, schema: Schema) -> nn.Module:
        manual_seed(seed)
        model = resnet20_32x32(schema.get_num_classes(), self.batch_norm)
        if self.pretrained:
            state_dict = torch.load(file.parent / "resnet20-12fca82f.th")["state_dict"]
            # Remove prefix "module." from state dict keys if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            del state_dict["linear.weight"]
            del state_dict["linear.bias"]
            model.load_state_dict(state_dict, strict=False)
        return model
        


AnyModel = PerceptronArgs | ResNet_32x32Args
