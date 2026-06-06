from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal
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

    norm: Literal["Identity", "BatchNorm", "LayerNorm", "GroupNorm"] = "BatchNorm"
    pretrained: bool = True

    def build(self, seed: int, schema: Schema) -> nn.Module:
        manual_seed(seed)
        model = resnet20_32x32(schema.get_num_classes(), self.norm)
        if self.pretrained:
            state_dict = torch.load(file.parent / "resnet20-12fca82f.th")["state_dict"]
            # Remove prefix "module." from state dict keys if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            del state_dict["linear.weight"]
            del state_dict["linear.bias"]
            model.load_state_dict(state_dict, strict=False)
        return model


@dataclass
class ConvNeXtArgs(ModelArgs):
    type_: ClassVar[str] = "ConvNextV2"
    pretrained: bool = True

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.convnext import ConvNextV2

        manual_seed(seed)
        return ConvNextV2(
            num_classes=schema.get_num_classes(), pretrained=self.pretrained
        )


@dataclass
class AirbenchCNNArgs(ModelArgs):
    type_: ClassVar[str] = "AirbenchCNN"
    channels_per_group: int = 16

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.airbench import AirbenchCNN

        manual_seed(seed)
        return AirbenchCNN(
            num_classes=schema.get_num_classes(),
            channels_per_group=self.channels_per_group,
        )


@dataclass
class DeepLightweightMLPArgs(ModelArgs):
    type_: ClassVar[str] = "DeepLightweightMLP"

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.deep_lightweight_mlp import DeepLightweightMLP

        manual_seed(seed)
        return DeepLightweightMLP()


@dataclass
class FashionCNNArgs(ModelArgs):
    type_: ClassVar[str] = "FashionCNN"

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.fashion_cnn import FashionCNN

        manual_seed(seed)
        return FashionCNN(num_classes=schema.get_num_classes())


AnyModel = (
    PerceptronArgs
    | ResNet_32x32Args
    | ConvNeXtArgs
    | AirbenchCNNArgs
    | DeepLightweightMLPArgs
    | FashionCNNArgs
)
