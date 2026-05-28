from src.config import Config
import optuna
from typing import Callable
from src.experiment import Experiment
from loguru import logger
from src.util import obj_dot_notation_set


def recreate_study(
    *, study_name: str, storage: str, direction: str = "maximize"
) -> "optuna.study.Study":
    summaries = optuna.get_all_study_summaries(storage=storage)
    if any(summary.study_name == study_name for summary in summaries):
        logger.warning(
            "Deleting existing Optuna study '{}' from storage '{}'.",
            study_name,
            storage,
        )
        optuna.delete_study(study_name=study_name, storage=storage)

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=False,
    )


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
        self.study = recreate_study(
            study_name=config.study_name,
            storage=config.hpsearch.storage
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
        logger.info("{}", self.config)
        experiment = Experiment(self.config)
        metrics = experiment.run()

        trial.set_user_attr("accuracy_seen_avg", float(metrics.accuracy_seen_avg))
        trial.set_user_attr("accuracy_all_avg", float(metrics.accuracy_all_avg))
        trial.set_user_attr("accuracy_final", float(metrics.accuracy_final))
        trial.set_user_attr("logdir", str(experiment.logdir.as_posix()))
        return float(metrics.accuracy_all_avg)
