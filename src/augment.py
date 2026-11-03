"""AutoAugment for the flat, normalized image rows that CapyMOA passes to learners."""

import torch
import torchvision.transforms as T
from torch import nn


class FlatAutoAugment(nn.Module):
    """Apply AutoAugment (CIFAR-10 policy) to a batch of flat, normalized images.

    One random policy is drawn per call and applied to the whole batch.
    """

    def __init__(self, shape: tuple[int, int, int], mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.shape = shape
        self._augment = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
        self.register_buffer("mean", mean.reshape(1, -1, 1, 1).float(), persistent=False)
        self.register_buffer("std", std.reshape(1, -1, 1, 1).float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        img = x.float().view(-1, *self.shape) * self.std + self.mean
        img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        img = self._augment(img).float() / 255.0
        return ((img - self.mean) / self.std).reshape(x.shape).to(dtype)
