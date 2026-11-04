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


class FlatFlipCrop(nn.Module):
    """Random horizontal flip and random crop (zero padding) on flat, normalized images.

    Each image gets its own flip and offset. No rotation, so it is safe for rotated
    domains. Zero padding in normalized space fills with the channel mean.
    """

    def __init__(self, shape: tuple[int, int, int], padding: int = 4):
        super().__init__()
        self.shape = shape
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        c, h, w = self.shape
        img = x.view(n, c, h, w)
        flip = torch.rand(n, device=x.device) < 0.5
        img = torch.where(flip.view(n, 1, 1, 1), img.flip(-1), img)
        p = self.padding
        padded = torch.nn.functional.pad(img, (p, p, p, p))
        dy = torch.randint(0, 2 * p + 1, (n,), device=x.device)
        dx = torch.randint(0, 2 * p + 1, (n,), device=x.device)
        rows = (dy.view(n, 1) + torch.arange(h, device=x.device)).view(n, 1, h, 1)
        cols = (dx.view(n, 1) + torch.arange(w, device=x.device)).view(n, 1, 1, w)
        out = padded[
            torch.arange(n, device=x.device).view(n, 1, 1, 1),
            torch.arange(c, device=x.device).view(1, c, 1, 1),
            rows.expand(n, 1, h, w),
            cols.expand(n, 1, h, w),
        ]
        return out.reshape(x.shape)
