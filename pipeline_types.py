from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Strategy:
    """A strategy and whether it can use learned detectors."""

    name: str
    use_detector: bool

    def __str__(self) -> str:
        return self.name


class Boundary(Enum):
    ABRUPT = "00_abrupt"
    GRADUAL = "01_gradual"
    SLOW = "02_slow"

    def __str__(self) -> str:
        return self.value


class Detector(Enum):
    ADWIN = "ADWIN"
    CUSUM = "CUSUM"
    ORACLE = "oracle"

    def __str__(self) -> str:
        return self.value
