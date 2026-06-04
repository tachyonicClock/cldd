from dataclasses import dataclass, asdict
from typing import Dict, ClassVar, Sequence, Any
import optuna


class _SuggestBase:
    type_: ClassVar[str]

    def _suggest(self, name: str, trial: optuna.Trial) -> Any:
        raise NotImplementedError


@dataclass
class SuggestCategorical(_SuggestBase):
    type_: ClassVar[str] = "categorical"
    choices: Sequence[Any]

    def _suggest(self, name: str, trial: optuna.Trial) -> Any:
        return trial.suggest_categorical(name=name, choices=self.choices)


@dataclass
class SuggestInt(_SuggestBase):
    type_: ClassVar[str] = "int"
    low: int
    high: int
    step: int = 1
    log: bool = False

    def _suggest(self, name: str, trial: optuna.Trial) -> Any:
        return trial.suggest_int(name=name, **asdict(self))


@dataclass
class SuggestFloat(_SuggestBase):
    type_: ClassVar[str] = "float"
    low: float
    high: float

    step: float | None = None
    log: bool = False

    def _suggest(self, name: str, trial: optuna.Trial) -> float:
        return trial.suggest_float(name=name, **asdict(self))


SuggestAny = SuggestCategorical | SuggestInt | SuggestFloat


@dataclass
class HPSearchConfig:
    args: Dict[str, SuggestAny]

    n_trials: int
    study_prefix: str = "bocl"
    metric: str = "accuracy_forgetful"
    storage: str = "logs/optuna.journal.log"
    maximize: bool = True

    def suggest(
        self, trial: optuna.Trial, startswith: str | None = None
    ) -> Dict[str, Any]:
        suggestions = {}
        for name, suggest in self.args.items():
            if startswith is not None and not name.startswith(startswith):
                continue
            suggestions[name] = suggest._suggest(name, trial)
        return suggestions
