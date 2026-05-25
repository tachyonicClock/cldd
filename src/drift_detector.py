from typing import ClassVar
from capymoa.drift.base_detector import BaseDriftDetector
from dataclasses import dataclass


@dataclass
class DriftDetectorArgs:
    type_: ClassVar[str]

    def build(self, seed: int) -> "BaseDriftDetector":
        raise NotImplementedError(
            "Must implement build method for drift detector config"
        )


@dataclass
class ADWINArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "ADWIN"


@dataclass
class EDDMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "EDDM"


@dataclass
class DDMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "DDM"


@dataclass
class OracleArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "oracle"


AnyDriftDetector = ADWINArgs | EDDMArgs | DDMArgs | OracleArgs
