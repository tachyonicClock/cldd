from collections.abc import Callable
from copy import deepcopy

import optuna
from loguru import logger

from src.config import Config
from src.experiment import Experiment
from src.util import obj_dot_notation_set


def optimize_with_max_trials(
    study: "optuna.study.Study",
    objective: Callable[[optuna.trial.Trial], tuple[float, ...] | float],
    n_trials: int,
    states: tuple[optuna.trial.TrialState, ...] = (optuna.trial.TrialState.COMPLETE,),
    callbacks=None,
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

    if callbacks is None:
        callbacks = []
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
        self.config = config
        self.hpsearch = config.hpsearch
        if self.hpsearch.args is None:
            logger.warning("Running only 1 trial since the search space is empty.")
            self.hpsearch.n_trials = 1
        self.study = self.hpsearch.new_study(config.study_name, config.dataset_root)

    def optimize(self) -> None:
        states = (optuna.trial.TrialState.COMPLETE,)
        if self.hpsearch.pruner != "none":
            # Pruned trials count toward the budget too
            states += (optuna.trial.TrialState.PRUNED,)
        optimize_with_max_trials(
            self.study,
            self._objective,
            n_trials=self.hpsearch.n_trials,
            states=states,
            n_jobs=self.hpsearch.n_jobs,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        # Apply suggestions to config.
        config = deepcopy(self.config)
        suggestions = self.hpsearch.suggest(trial)
        for key, value in suggestions.items():
            obj_dot_notation_set(key, config, value)

        logger.info(f"Trial {trial.number} with suggestions:")
        config.trial = trial.number
        config.seed = trial.number
        experiment = Experiment(config)
        handlers = []
        if self.hpsearch.pruner != "none":
            from src.pruning import PruneOnSeenAccuracy

            handlers.append(PruneOnSeenAccuracy(trial))
        metrics = experiment.run(handlers=handlers)

        trial.set_user_attr("accuracy_seen_avg", float(metrics["accuracy_seen_avg"]))
        trial.set_user_attr("accuracy_all_avg", float(metrics["accuracy_all_avg"]))
        trial.set_user_attr("accuracy_final", float(metrics["accuracy_final"]))
        trial.set_user_attr("logdir", str(experiment.logdir.as_posix()))
        return float(metrics[self.hpsearch.metric])
