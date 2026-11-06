"""ResNet-18 for small images: ImageNet-pretrained and fine-tuned, or trained from scratch."""

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
        norm: str = "batch",
        cifar_stem: bool = False,
        base_width: int = 64,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        if base_width != 64 and pretrained:
            raise ValueError("Only base_width 64 has pretrained weights.")
        if norm == "group":
            if pretrained:
                raise ValueError("GroupNorm has no pretrained weights; use pretrained=False.")
            # No running statistics to drift between domains. Groups of 32 as in Wu and
            # He (2018), or fewer when the width is too small to divide.
            norm_layer = lambda c: nn.GroupNorm(32 if c % 32 == 0 else c // 4, c)
        else:
            norm_layer = None
        self.net = resnet18(weights=weights, norm_layer=norm_layer)
        if base_width != 64:
            # Reduced ResNet-18 (Lopez-Paz and Ranzato, 2017): same layout, fewer maps.
            from torchvision.models.resnet import BasicBlock, ResNet
            self.net = ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=norm_layer)
            self.net.inplanes = base_width
            w = base_width
            self.net.conv1 = nn.Conv2d(3, w, kernel_size=7, stride=2, padding=3, bias=False)
            self.net.bn1 = (norm_layer or nn.BatchNorm2d)(w)
            self.net.layer1 = self.net._make_layer(BasicBlock, w, 2)
            self.net.layer2 = self.net._make_layer(BasicBlock, 2 * w, 2, stride=2)
            self.net.layer3 = self.net._make_layer(BasicBlock, 4 * w, 2, stride=2)
            self.net.layer4 = self.net._make_layer(BasicBlock, 8 * w, 2, stride=2)
            self.net.fc = nn.Linear(8 * w, num_classes)
        if cifar_stem:
            # 3x3 stride-1 stem without max pooling, the usual change for 32x32 inputs.
            self.net.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()
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
