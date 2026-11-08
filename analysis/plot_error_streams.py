"""Generate an empty placeholder for the error-streams figure."""

import matplotlib.pyplot as plt
from plots import save_fig


def main() -> None:
    fig, _ax = plt.subplots()
    save_fig(fig, "error_streams.pdf")


if __name__ == "__main__":
    main()
