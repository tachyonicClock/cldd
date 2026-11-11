from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

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
    args: dict[str, SuggestAny] | None = None

    n_trials: int = 1
    n_jobs: int = 1
    study_prefix: str = "bocl"
    metric: str = "accuracy_final"
    storage: str | None = None
    """Optuna journal file. None uses the journal of the dataset (see `new_study`)."""
    maximize: bool = True

    multivariate: bool = False
    """Use multivariate TPE, which models the hyperparameters jointly."""
    constant_liar: bool = False
    """TPE avoids proposing the same region to parallel workers (n_jobs > 1)."""
    pruner: Literal["none", "median"] = "none"
    """Prune trials whose running score falls below the median of earlier trials."""
    pruner_warmup_steps: int = 2
    """Tasks before pruning can start. Regularisers only act after the first task."""
    pruner_startup_trials: int = 10
    """Trials that always run to the end before pruning starts."""

    mode: Literal["w", "a", "x"] = "x"
    """Control what happens if the study already exists. "w" to overwrite, "a" to
    append, "x" to raise an error."""

    def new_study(self, study_name: str, dataset_root: Path) -> optuna.Study:
        journal = Path(self.storage or dataset_root / "optuna.journal.log")
        journal.parent.mkdir(parents=True, exist_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal.as_posix())  # type: ignore
        )

        if self.mode == "w":
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except KeyError:
                pass

        sampler = None
        if self.multivariate or self.constant_liar:
            sampler = optuna.samplers.TPESampler(
                multivariate=self.multivariate, constant_liar=self.constant_liar
            )
        pruner = None
        if self.pruner == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.pruner_startup_trials,
                n_warmup_steps=self.pruner_warmup_steps,
            )
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize" if self.maximize else "minimize",
            load_if_exists=self.mode == "a",
            sampler=sampler,
            pruner=pruner,
        )

    def suggest(
        self, trial: optuna.Trial, startswith: str | None = None
    ) -> dict[str, Any]:
        suggestions = {}
        if self.args is None:
            return suggestions
        for name, suggest in self.args.items():
            if startswith is not None and not name.startswith(startswith):
                continue
            suggestions[name] = suggest._suggest(name, trial)
        return suggestions
