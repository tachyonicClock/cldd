#!/home/antonlee/.local/bin/uv run
import os
import subprocess as sp
from typing import Sequence
from loguru import logger
from pathlib import Path
import click
import time

N_SUBPROCESSES = 2
TIMEOUT = 10

logs = Path("logs") / "console"


# Use -G to specify GPU
@click.command()
@click.option("-g", "--gpu", multiple=True, type=int, help="GPU(s) to use")
@click.option(
    "-n",
    "--n-subprocesses",
    default=N_SUBPROCESSES,
    type=int,
    help="Number of subprocesses to run",
)
def main(gpu: Sequence[int], n_subprocesses: int):
    # Setup nvidia-mps
    logs.mkdir(parents=True, exist_ok=True)

    server_env = {}
    if gpu:
        server_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu)

    client_env = {
        "PATH": os.environ["PATH"],
        "CAPYMOA_DATASETS_DIR": os.environ.get("CAPYMOA_DATASETS_DIR", ""),
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
        # nvidia-mps remaps GPU IDs to a contiguous range starting from 0, so we
        # need to set this for the clients as well
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(len(gpu))),
    }

    logger.info(f"SERVER ENV: {server_env}")
    logger.info(f"CLIENT ENV: {client_env}")

    with open(logs / "mps_server.log", "w") as server_log:
        server = sp.Popen(
            ["nvidia-cuda-mps-control", "-f"],
            stdin=sp.PIPE,
            stdout=server_log,
            stderr=sp.STDOUT,
            env=server_env,
        )
        time.sleep(5)

    try:
        # Run the main script
        sp.run(
            ["uv", "run", "doit", "-n", str(n_subprocesses)],
            check=True,
            env=client_env,
        )
    finally:
        logger.warning("Cleaning up nvidia-mps...")
        # Clean up nvidia-mps
        client = sp.Popen(["nvidia-cuda-mps-control"], stdin=sp.PIPE, env=client_env)
        try:
            client.communicate(input=b"quit\n", timeout=TIMEOUT)
            client.wait(TIMEOUT)
            server.wait(TIMEOUT)
        except sp.TimeoutExpired:
            logger.error(
                "Timeout expired while waiting for nvidia-mps processes to exit. Forcing termination."
            )
            client.kill()
            server.kill()


if __name__ == "__main__":
    main()
