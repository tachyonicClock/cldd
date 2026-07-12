from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, ClassVar, Literal, Optional, Sequence, Any
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
    args: Optional[Dict[str, SuggestAny]] = None

    n_trials: int = 1
    n_jobs: int = 1
    study_prefix: str = "bocl"
    metric: str = "accuracy_final"
    storage: str = "logs/optuna.journal.log"
    maximize: bool = True

    mode: Literal["w", "a", "x"] = "x"
    """Control what happens if the study already exists. "w" to overwrite, "a" to
    append, "x" to raise an error."""

    def new_study(self, study_name: str) -> optuna.Study:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(Path(self.storage).as_posix())  # type: ignore
        )

        if self.mode == "w":
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                pass

        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize" if self.maximize else "minimize",
            load_if_exists=self.mode == "a",
        )

    def suggest(
        self, trial: optuna.Trial, startswith: str | None = None
    ) -> Dict[str, Any]:
        suggestions = {}
        if self.args is None:
            return suggestions
        for name, suggest in self.args.items():
            if startswith is not None and not name.startswith(startswith):
                continue
            suggestions[name] = suggest._suggest(name, trial)
        return suggestions
