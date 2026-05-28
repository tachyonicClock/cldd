from types_ import Strategy, Detector, Boundary
from pathlib import Path
from typing import Sequence
from loguru import logger


def tune_strategy_hp(
    target: Path,
    base_config: Path,
    strategy: Strategy,
    detector: Detector,
    boundary: Boundary,
):
    logger.info(f"Tuning strategy {strategy} with detector {detector} and boundary {boundary}")
    target.touch()


def error_streams(
    target: Path,
    strategy_config: Path,
    strategy: Strategy,
    detector: Detector,
    boundary: Boundary,
    seed: int,
):
    logger.info(f"Generating error streams for strategy {strategy} with detector {detector} and boundary {boundary} and seed {seed}")
    target.touch()


def tune_hp_detector(
    target: Path,
    error_streams: Sequence[Path],
    strategy: Strategy,
    detector: Detector,
    boundary: Boundary,
):
    pass


def select_best_detector(
    target: Path,
    detector_configs: Sequence[Path],
    strategy: Strategy,
    boundary: Boundary,
):
    pass

def evaluate(
    target: Path,
    strategy: Strategy,
    strategy_config: Path,
    detector: Detector,
    detector_config: Path,
    boundary: Boundary,
):
    pass