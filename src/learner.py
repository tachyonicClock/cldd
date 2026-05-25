from dataclasses import dataclass
from typing import ClassVar
from capymoa.base import BatchClassifier
from torch import nn


@dataclass
class LearnerArgs:
    type_: ClassVar[str]

    def build(self, seed: int, model: nn.Module) -> BatchClassifier:
        raise NotImplementedError("Must implement build method for learner config")


@dataclass
class EWCArgs(LearnerArgs):
    type_: ClassVar[str] = "EWC"


@dataclass
class SIArgs(LearnerArgs):
    type_: ClassVar[str] = "SI"


@dataclass
class LWFArgs(LearnerArgs):
    type_: ClassVar[str] = "LWF"


@dataclass
class DERArgs(LearnerArgs):
    type_: ClassVar[str] = "DER"


@dataclass
class PNArgs(LearnerArgs):
    type_: ClassVar[str] = "PN"


@dataclass
class RARArgs(LearnerArgs):
    type_: ClassVar[str] = "RAR"


@dataclass
class ERArgs(LearnerArgs):
    type_: ClassVar[str] = "ER"


@dataclass
class FTArgs(LearnerArgs):
    type_: ClassVar[str] = "FT"


AnyLearner = EWCArgs | SIArgs | LWFArgs | DERArgs | PNArgs | RARArgs | ERArgs | FTArgs
