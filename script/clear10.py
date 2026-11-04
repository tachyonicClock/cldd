"""Export CLEAR10 to a compact .pt file of uint8 images.

Each of the 10 labelled train buckets (time periods) is a task, in time order. Test
images come from the matching test buckets. Bucket 0 has no train images and is
skipped. Images are centre-cropped to a square and resized.
"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from os import environ
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BUCKETS = range(1, 11)


def load_image(args: tuple[str, int]) -> np.ndarray:
    path, size = args
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)


def load_split(root: Path, classes: list[str], size: int, pool: ProcessPoolExecutor):
    tasks = []
    for bucket in BUCKETS:
        paths, labels = [], []
        for label, name in enumerate(classes):
            files = sorted((root / str(bucket) / name).iterdir())
            paths += [str(f) for f in files]
            labels += [label] * len(files)
        images = list(pool.map(load_image, [(p, size) for p in paths], chunksize=64))
        tasks.append((torch.from_numpy(np.stack(images)), torch.tensor(labels)))
        print(f"{root.parent.name} bucket {bucket}: {len(labels)} images")
    return tasks


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32, help="Image side in pixels")
    parser.add_argument("--workers", type=int, default=4, help="Decoding processes")
    parser.add_argument("--root", type=Path, default=Path(environ["DATASETS"]) / "clear10")
    args = parser.parse_args()

    classes = (args.root / "train" / "class_names.txt").read_text().split()
    out = Path(__file__).resolve().parent.parent / "data" / f"clear10_{args.size}.pt"
    with ProcessPoolExecutor(args.workers) as pool:
        data = {
            "classes": classes,
            "train": load_split(args.root / "train" / "labeled_images", classes, args.size, pool),
            "test": load_split(args.root / "test" / "labeled_images", classes, args.size, pool),
        }
    torch.save(data, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
