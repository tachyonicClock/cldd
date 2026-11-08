"""Generate an empty placeholder for the drift-speed figure."""

import matplotlib.pyplot as plt
from plots import save_fig


def main() -> None:
    fig, _ax = plt.subplots()
    save_fig(fig, "drift_speed.pdf")


if __name__ == "__main__":
    main()
