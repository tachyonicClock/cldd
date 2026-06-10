#!/home/antonlee/.local/bin/uv run
import os
from signal import signal
import subprocess as sp
from typing import Sequence
from loguru import logger
from pathlib import Path
import click
import time
import sys
import random

N_SUBPROCESSES = 2
TIMEOUT = 10

logs = Path("logs") / "console"

def get_free_GPUs() -> Sequence[str]:
    return sp.check_output(
        "comm -23 "
        "<(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | sort) "
        "<(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)",
        shell=True,
        text=True,
    ).splitlines()


# Use -G to specify GPU
@click.command()
@click.option(
    "-n",
    "--n-subprocesses",
    default=N_SUBPROCESSES,
    type=int,
    help="Number of subprocesses to run",
)
@click.option(
    "-r",
    "--run",
    default=[],
    multiple=True,
    type=str,
    help="Name of the run to execute",
)
def main(n_subprocesses: int, run: Sequence[str]):
    # Setup nvidia-mps
    logs.mkdir(parents=True, exist_ok=True)
    free_gpu = get_free_GPUs()
    if len(free_gpu) == 0:
        logger.error("No free GPUs found. Exiting.")
        sys.exit(1)
    gpu = random.choice(free_gpu)

    pid = os.getpid()
    pgid = os.getpgid(pid)
    logger.info(f"Stop me and my children with: `kill -TERM -- -{pgid}`")

    server_env = {
        "CUDA_VISIBLE_DEVICES": gpu,
    }
    client_env = {
        "PATH": os.environ["PATH"],
        "CAPYMOA_DATASETS_DIR": os.environ.get("CAPYMOA_DATASETS_DIR", ""),
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
        # nvidia-mps remaps GPU IDs to a contiguous range starting from 0, so we
        # need to set this for the clients as well
        "CUDA_VISIBLE_DEVICES": gpu,
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
        cmd = ["uv", "run", "python", "-u", "-m", "doit", "-n", str(n_subprocesses)]
        if run:
            cmd.extend(run)
        sp.run(
            cmd,
            env=client_env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    finally:
        logger.warning("Cleaning up nvidia-mps...")
        server.terminate()
        try:
            server.wait(timeout=TIMEOUT)
        finally:
            if server.poll() is None:
                logger.error("MPS server did not terminate gracefully, killing...")
                server.kill()


if __name__ == "__main__":
    main()
