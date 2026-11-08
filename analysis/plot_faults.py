"""Generate an empty placeholder for the faults figure."""

import matplotlib.pyplot as plt
from plots import save_fig


def main() -> None:
    fig, _ax = plt.subplots()
    save_fig(fig, "faults.pdf")


if __name__ == "__main__":
    main()
