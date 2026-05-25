from dataclasses import dataclass
from capymoa.stream import Schema
from typing import ClassVar
from capymoa.base import BatchClassifier
from torch import nn
from capymoa.classifier import Finetune
from torch.optim import Adam


@dataclass
class LearnerArgs:
    type_: ClassVar[str]
    lr: float = 0.001

    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
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

    def build(
        self, seed: int, schema: Schema, device: str, model: nn.Module
    ) -> BatchClassifier:
        return Finetune(
            schema,
            model,
            optimizer=Adam(model.parameters(), lr=self.lr),
            device=device,
            random_seed=seed,
        )


AnyLearner = EWCArgs | SIArgs | LWFArgs | DERArgs | PNArgs | RARArgs | ERArgs | FTArgs
