"""

Based on:
* https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench/lib_airbench93.py

Notes:
* Remove whitening layer. As we don't implement the same setup.
* Replace BatchNorm with GroupNorm, as the former is less stable with small batch sizes
  and when domain shifts are present.

"""

import torch
from torch import Tensor, nn


class Conv(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str | int = "same",
        bias: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        channels_per_group: int,
    ):
        super().__init__()
        num_groups = channels_out // channels_per_group

        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.GroupNorm(num_groups, channels_out, affine=False)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = nn.GroupNorm(num_groups, channels_out, affine=False)
        self.activ = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class AirbenchCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels_per_group: int = 8,
    ):
        super().__init__()
        width_block1 = 64
        width_block2 = 128
        width_block3 = 128

        self.features = nn.Sequential(
            ConvGroup(3, width_block1, channels_per_group),
            ConvGroup(width_block1, width_block2, channels_per_group),
            ConvGroup(width_block2, width_block3, channels_per_group),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(width_block3, num_classes, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))
