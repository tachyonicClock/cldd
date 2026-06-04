"""

Based on:
* https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench/lib_airbench93.py

Notes:
* Remove whitening layer. As we don't implement the same setup.
* Replace BatchNorm with GroupNorm, as the former is less stable with small batch sizes
  and when domain shifts are present.

"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class Mul(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


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
        groupnorm_groups: int,
    ):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = nn.GroupNorm(groupnorm_groups, channels_out, affine=False)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = nn.GroupNorm(groupnorm_groups, channels_out, affine=False)
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
        widths: Mapping[str, int] | None = None,
        groupnorm_groups: int = 8,
        scaling_factor: float = 1 / 9,
    ):
        super().__init__()
        if widths is None:
            widths = {
                "block1": 64,
                "block2": 128,
                "block3": 128,
            }

        self.net = nn.Sequential(
            ConvGroup(3, widths["block1"], groupnorm_groups),
            ConvGroup(widths["block1"], widths["block2"], groupnorm_groups),
            ConvGroup(widths["block2"], widths["block3"], groupnorm_groups),
            nn.MaxPool2d(3),
            Flatten(),
            nn.Linear(widths["block3"], num_classes, bias=False),
            Mul(scaling_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
