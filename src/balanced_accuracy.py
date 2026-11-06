"""Balanced accuracy (mean per-class recall) per train/test task, robust to skewed labels."""

import numpy as np
from capymoa.ocl.evaluation.events import EvalBatchPredict
from capymoa.ocl.events import Dispatcher, Handler


class BalancedAccuracy(Handler):
    """Collects per-class hits for each (train task, test task) pair during evaluation."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self._hit: dict[tuple[int, int], np.ndarray] = {}
        self._total: dict[tuple[int, int], np.ndarray] = {}

    def on_eval_batch(self, event: EvalBatchPredict):
        key = (event.train_task, event.test_task)
        y, y_hat = event.y.cpu().numpy(), event.y_hat.cpu().numpy()
        hit = np.bincount(y[y == y_hat], minlength=self.n_classes)
        total = np.bincount(y, minlength=self.n_classes)
        self._hit[key] = self._hit.get(key, 0) + hit
        self._total[key] = self._total.get(key, 0) + total

    def matrix(self, n_tasks: int) -> np.ndarray:
        """`m[i, j]` is the balanced accuracy on test task j after training on task i."""
        m = np.full((n_tasks, n_tasks), np.nan)
        for (i, j), total in self._total.items():
            present = total > 0
            m[i, j] = (self._hit[(i, j)][present] / total[present]).mean()
        return m

    def summary(self, n_tasks: int) -> dict[str, float]:
        m = self.matrix(n_tasks)
        last = n_tasks - 1
        return {
            "balanced_accuracy_seen_avg": float(
                np.mean([np.nanmean(m[i, : i + 1]) for i in range(n_tasks)])
            ),
            "balanced_accuracy_final": float(np.nanmean(m[last])),
            "balanced_backward_transfer": float(
                np.nanmean([m[last, j] - m[j, j] for j in range(last)])
            ),
        }

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        dispatcher.subscribe(EvalBatchPredict, self.on_eval_batch)
        return self
