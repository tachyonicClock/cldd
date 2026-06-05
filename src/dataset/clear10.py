from torchvision.datasets import ImageFolder
from typing import Literal, Sequence
from pathlib import Path


def load_clear10(
    root: Path, split: Literal["train", "test"], transform
) -> Sequence[ImageFolder]:
    split_dir = root / "clear10_224" / split
    return [
        ImageFolder(split_dir / f"task_{task}", transform=transform)
        for task in range(10)
    ]
