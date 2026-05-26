from src.config import Config
import optuna
from typing import Any, Callable
from src.experiment import Experiment
from loguru import logger
from pprint import pprint


def obj_dot_notation_set(key: str, obj: object, value: Any) -> object:
    root = obj
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return root


def optimize_with_max_trials(
    study: "optuna.study.Study",
    objective: Callable[[optuna.trial.Trial], tuple[float, ...] | float],
    n_trials: int,
    states: tuple[optuna.trial.TrialState, ...] = (optuna.trial.TrialState.COMPLETE,),
    callbacks=[],
    **kwargs,
):
    """
    Run an Optuna study until the total number of completed trials reaches
    ``n_trials``.

    In parallel execution, Optuna treats ``n_trials`` as a per-worker limit.
    This helper enforces a study-wide cap by:

    - counting trials already finished in ``states``
    - returning early if the target has already been reached
    - attaching ``MaxTrialsCallback`` so concurrent workers stop when the
      shared limit is reached

    This avoids scheduling extra work when using multiple processes and
    preserves progress across restarts by including previously completed trials
    in the count.

    Source: https://github.com/optuna/optuna/issues/1883#issuecomment-702688136
    """

    trials = study.get_trials(deepcopy=False, states=states)
    n_complete = len(trials)

    if n_complete >= n_trials:
        return

    callbacks.append(optuna.study.MaxTrialsCallback(n_trials, states=states))

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        # catch=[],
        **kwargs,
    )


class HPSearch:
    def __init__(self, config: Config) -> None:
        assert config.hpsearch is not None
        study_name = "/".join(
            [
                config.hpsearch.study_prefix,
                config.label,
                config.scenario_label,
                config.method_label,
            ]
        )
        logger.info(f"Setup study `{study_name}`.")
        self.study = optuna.create_study(
            study_name=study_name,
            storage=config.hpsearch.storage,
            direction="maximize",
            load_if_exists=True,
        )
        self.config = config
        self.hpsearch = config.hpsearch

    def optimize(self) -> None:
        optimize_with_max_trials(
            self.study,
            self._objective,
            n_trials=self.hpsearch.n_trials,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        # Apply suggestions to config.
        suggestions = self.hpsearch.suggest(trial)
        for key, value in suggestions.items():
            obj_dot_notation_set(key, self.config, value)

        logger.info(f"Trial {trial.number} with suggestions:")
        self.config.trial = trial.number
        self.config.seed = trial.number
        pprint(self.config)
        experiment = Experiment(self.config)
        metrics = experiment.run()

        trial.set_user_attr("accuracy_seen_avg", metrics.accuracy_seen_avg)
        trial.set_user_attr("accuracy_all_avg", metrics.accuracy_all_avg)
        trial.set_user_attr("accuracy_final", metrics.accuracy_final)
        return metrics.accuracy_all_avg
