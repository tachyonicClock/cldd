"""Report a running continual-learning score to Optuna so weak trials can be pruned."""

import optuna
from capymoa.ocl.evaluation.events import EvalBatchPredict, TestBegin, TestEnd
from capymoa.ocl.events import Dispatcher, Handler


class PruneOnSeenAccuracy(Handler):
    """After each task's evaluation, report the running mean of accuracy on seen tasks.

    The running mean after the last task is the tuning objective (accuracy_seen_avg),
    so earlier values are partial versions of it. Raises ``optuna.TrialPruned`` when
    the trial's pruner says so.
    """

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self._history: list[float] = []
        self._correct: dict[int, int] = {}
        self._total: dict[int, int] = {}
        self._train_task: int | None = None

    def on_test_begin(self, event: TestBegin):
        self._correct.clear()
        self._total.clear()
        self._train_task = None

    def on_eval_batch(self, event: EvalBatchPredict):
        self._train_task = event.train_task
        j = event.test_task
        self._correct[j] = self._correct.get(j, 0) + int(
            (event.y_hat.cpu() == event.y.cpu()).sum()
        )
        self._total[j] = self._total.get(j, 0) + len(event.y)

    def on_test_end(self, event: TestEnd):
        t = self._train_task
        if t is None:
            return
        seen = [
            self._correct[j] / self._total[j]
            for j in range(t + 1)
            if self._total.get(j)
        ]
        self._history.append(sum(seen) / len(seen))
        value = sum(self._history) / len(self._history)
        self.trial.report(value, step=t)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned after task {t + 1} at {value:.4f}")

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        dispatcher.subscribe(TestBegin, self.on_test_begin)
        dispatcher.subscribe(EvalBatchPredict, self.on_eval_batch)
        dispatcher.subscribe(TestEnd, self.on_test_end)
        return self
