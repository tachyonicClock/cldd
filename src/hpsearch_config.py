from dataclasses import dataclass, asdict
from typing import Dict, ClassVar, Sequence, Any
import optuna
import os


class _SuggestBase:
    type_: ClassVar[str]

    def _suggest(self, name: str, trial: optuna.Trial) -> None:
        raise NotImplementedError


@dataclass
class SuggestCategorical(_SuggestBase):
    type_: ClassVar[str] = "categorical"
    choices: Sequence[Any]

    def _suggest(self, name: str, trial: optuna.Trial) -> None:
        trial.suggest_categorical(name=name, choices=self.choices)


@dataclass
class SuggestInt(_SuggestBase):
    type_: ClassVar[str] = "int"
    low: int
    high: int
    step: int = 1
    log: bool = False

    def _suggest(self, name: str, trial: optuna.Trial) -> None:
        trial.suggest_int(name=name, **asdict(self))


@dataclass
class SuggestFloat(_SuggestBase):
    type_: ClassVar[str] = "float"
    low: float
    high: float

    step: float | None = None
    log: bool = False

    def _suggest(self, name: str, trial: optuna.Trial) -> None:
        trial.suggest_float(name=name, **asdict(self))


SuggestAny = SuggestCategorical | SuggestInt | SuggestFloat


@dataclass
class HPSearchConfig:
    args: Dict[str, SuggestAny]
    storage: str = os.environ.get("OPTUNA_STORAGE", "sqlite:///optuna.db")

    def suggest(self, trial: optuna.Trial) -> None:
        for name, suggest in self.args.items():
            suggest._suggest(name, trial)
