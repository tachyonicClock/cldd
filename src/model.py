from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal
from torch import nn
from capymoa.stream import Schema
from capymoa.core.torch.ann import Perceptron, resnet20_32x32
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
    hidden_size: int = 50

    def build(self, seed: int, schema: Schema) -> nn.Module:
        manual_seed(seed)
        return Perceptron(schema, hidden_size=self.hidden_size)


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
    width: int = 1
    """Multiplies the channels of every block."""

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.airbench import AirbenchCNN

        manual_seed(seed)
        return AirbenchCNN(
            num_classes=schema.get_num_classes(),
            channels_per_group=self.channels_per_group,
            width=self.width,
        )


@dataclass
class IncrementalCNNArgs(ModelArgs):
    type_: ClassVar[str] = "IncrementalCNN"

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.incremental_cnn import IncrementalCNN

        manual_seed(seed)
        return IncrementalCNN(
            num_classes=schema.get_num_classes(),
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


@dataclass
class PretrainedResNet18Args(ModelArgs):
    type_: ClassVar[str] = "ResNet18"
    image_size: int = 64
    """Inputs are upsampled to this size before the network."""
    pretrained: bool = True
    freeze_bn_stats: bool = True
    """Keep ImageNet BatchNorm statistics; scale and shift still train."""
    norm: Literal["batch", "group"] = "batch"
    """GroupNorm (from scratch only) has no running statistics to drift between domains."""
    cifar_stem: bool = False
    """3x3 stride-1 stem without max pooling, for 32x32 inputs at image_size 32."""
    base_width: int = 64
    """Channels of the first stage; 20 gives the reduced ResNet-18 of GEM."""

    def build(self, seed: int, schema: Schema) -> nn.Module:
        from .models.resnet import PretrainedResNet18

        manual_seed(seed)
        return PretrainedResNet18(
            num_classes=schema.get_num_classes(),
            image_size=self.image_size,
            pretrained=self.pretrained,
            freeze_bn_stats=self.freeze_bn_stats,
            norm=self.norm,
            cifar_stem=self.cifar_stem,
            base_width=self.base_width,
        )


AnyModel = (
    PerceptronArgs
    | ResNet_32x32Args
    | ConvNeXtArgs
    | AirbenchCNNArgs
    | IncrementalCNNArgs
    | DeepLightweightMLPArgs
    | FashionCNNArgs
    | PretrainedResNet18Args
)
