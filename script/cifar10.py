"""Export CIFAR-10 from the Hugging Face arrow cache to a compact .pt file of uint8 images.

`RotatedCIFAR10` in `src/scenario.py` reads this file and rotates each task on the fly.
"""

import io
from argparse import ArgumentParser
from os import environ
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch
from PIL import Image

ARROW_DIR = "huggingface/uoft-cs___cifar10/plain_text/0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2"


def load_split(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    table = ipc.open_stream(str(path)).read_all()
    images = [
        np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"), dtype=np.uint8)
        for img in table.column("img").to_pylist()
    ]
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(table.column("label").to_pylist())
    return x, y


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(environ["DATASETS"]) / ARROW_DIR)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent.parent / "data" / "cifar10.pt"
    data = {split: load_split(args.root / f"cifar10-{split}.arrow") for split in ["train", "test"]}
    for split, (x, y) in data.items():
        print(split, tuple(x.shape), np.bincount(y.numpy()).tolist())
    torch.save(data, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
