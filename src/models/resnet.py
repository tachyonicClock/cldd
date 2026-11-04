"""ImageNet-pretrained ResNet-18 for small images, fine-tuned end to end."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class PretrainedResNet18(nn.Module):
    """ResNet-18 with ImageNet weights and a new linear head.

    Inputs arrive as flat rows of square 3-channel images and are upsampled to
    `image_size` before the network. With `freeze_bn_stats`, BatchNorm keeps its
    ImageNet running statistics (its scale and shift still train), so the statistics
    do not drift between domains, a source of forgetting that weight penalties
    such as EWC cannot prevent.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int = 64,
        pretrained: bool = True,
        freeze_bn_stats: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.image_size = image_size
        self.freeze_bn_stats = freeze_bn_stats

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            side = int(round((x.shape[1] / 3) ** 0.5))
            x = x.view(-1, 3, side, side)
        if x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=self.image_size, mode="bilinear", align_corners=False
            )
        return self.net(x)
