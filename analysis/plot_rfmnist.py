"""Draw the RotatedFashionMNIST distributions: one 3x3 grid of samples per distribution.

Run with `uv run analysis/plot_rfmnist.py` from the repo root.
The panels are the saved sample grids in `data/rfmnist_panels/`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from common import FONT_SUBTITLE, NO_TOP_RIGHT, PAPER_STYLE

plt.style.use(PAPER_STYLE)
plt.rcParams.update(NO_TOP_RIGHT)

PANEL_DIR = Path(__file__).resolve().parent.parent / "data" / "rfmnist_panels"
FIG_DIR = Path("fig")

panels = sorted(PANEL_DIR.glob("distribution_*.png"))
# One row, a little under the 302 pt of the original so the saved PDF fits
fig, axes = plt.subplots(1, len(panels), figsize=(302 / 72, 67 / 72), layout="constrained")
for i, (ax, path) in enumerate(zip(axes, panels), start=1):
    ax.imshow(Image.open(path), cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Distribution {i}", fontsize=FONT_SUBTITLE)
    ax.axis("off")
FIG_DIR.mkdir(exist_ok=True)
fig.savefig(FIG_DIR / "rotated_fashion_mnist_tasks.pdf")
