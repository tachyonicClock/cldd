from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator

from capymoa.ocl import datasets
import torch
from tqdm import tqdm


def image_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    min_value = float(tensor.min())
    max_value = float(tensor.max())
    if min_value < 0.0 or max_value > 1.0:
        raise ValueError(
            f"Tensor values should be in [0, 1], but found min={min_value} max={max_value}"
        )
    return (tensor * 255).to(torch.uint8)


def collect_tasks(
    loaders: Iterator, split_name: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tasks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for task in tqdm(loaders, desc=f"{split_name} tasks"):
        task_x: list[torch.Tensor] = []
        task_y: list[torch.Tensor] = []
        for batch_x, batch_y in tqdm(task, desc=f"{split_name} batches", leave=False):
            task_x.append(image_to_uint8(batch_x))
            task_y.append(batch_y)
        tasks.append((torch.cat(task_x), torch.cat(task_y)))
    return tasks


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Export RotatedFashionMNIST to a compact .pt file"
    )
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=128, help="Loader batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Loader workers")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "data"
        / "rotated_fashion_mnist_data.pt",
        help="Output .pt path",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    dataset = datasets.RotatedFashionMNIST(
        num_tasks=args.num_tasks,
        seed=args.seed,
        preload_test=False,
    )

    data = {
        "train": collect_tasks(
            dataset.train_loaders(args.batch_size, num_workers=args.num_workers),
            split_name="train",
        ),
        "test": collect_tasks(
            dataset.test_loaders(args.batch_size, num_workers=args.num_workers),
            split_name="test",
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
