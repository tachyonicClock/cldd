import numpy as np
from typing import Sequence
from dataclasses import dataclass


@dataclass
class DriftConfusion:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


def drift_confusion(
    trues: np.ndarray | Sequence[int],
    preds: np.ndarray | Sequence[int],
    max_early: int,
    max_delay: int,
) -> DriftConfusion:
    trues = np.sort(np.array(trues, dtype=int))
    preds = np.sort(np.array(preds, dtype=int))

    assert trues.ndim == 1 and preds.ndim == 1
    assert max_early >= 0 and max_delay >= 0

    if len(preds) == 0:
        return DriftConfusion(tp=0, fp=0, fn=len(trues))
    if len(trues) == 0:
        return DriftConfusion(tp=0, fp=len(preds), fn=0)

    tp, fp, fn = 0, 0, 0

    # Iterate over partitions defined by the midpoints between true drift points
    midpoints = (trues[:-1] + trues[1:]) / 2
    starts = np.concatenate(([-np.inf], midpoints))
    ends = np.concatenate((midpoints, [np.inf]))
    for true, start, end in zip(trues, starts, ends, strict=True):
        # Use < on the upper bound to prevent duplicating a prediction on a midpoint
        mask = (preds >= start) & (preds < end)
        in_window = (preds[mask] >= true - max_early) & (
            preds[mask] <= true + max_delay
        )
        in_count = int(np.sum(in_window))
        fp += int(np.sum(~in_window))  # Predictions outside window

        if in_count > 0:
            tp += 1
            fp += in_count - 1
        else:
            fn += 1
    return DriftConfusion(tp=tp, fp=fp, fn=fn)
