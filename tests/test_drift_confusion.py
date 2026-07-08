from capymoa.drift.eval_detector import EvaluateDriftDetector
from src.metrics import drift_confusion, DriftConfusion
import numpy as np
from typing import List, Sequence
from hypothesis import given, strategies as st
import pytest

def reference_drift_confusion(
    trues: np.ndarray | Sequence[int],
    preds: np.ndarray | Sequence[int],
    max_early: int,
    max_delay: int,
) -> DriftConfusion:
    evaluator = EvaluateDriftDetector(max_delay=max_delay, max_early_detection=max_early)
    metrics = evaluator.calc_performance(trues, preds, tot_n_instances=1)
    return DriftConfusion(
        metrics.tp,
        metrics.fp,
        metrics.fn
    )

@given(
    trues=st.lists(st.integers(min_value=0, max_value=100), unique=True),
    preds=st.lists(st.integers(min_value=0, max_value=100), unique=True),
    max_early=st.integers(min_value=0, max_value=10),
    max_delay=st.integers(min_value=1, max_value=10),
)
def test_drift_confusion_match_reference(
    trues: List[int], preds: List[int], max_early: int, max_delay: int
):
    trues.sort()
    preds.sort()
    result = drift_confusion(trues, preds, max_early, max_delay)
    assert result.tp + result.fp == len(preds), "tp + fp != len(preds)"
    assert result.tp + result.fn == len(trues), "tp + fn != len(trues)"
    

@pytest.mark.parametrize(
    "trues,preds,max_early,max_delay,expected",
    [
        ([3], [1, 5], 1, 1, DriftConfusion(tp=0, fp=2, fn=1)),
        ([3], [2, 4], 1, 1, DriftConfusion(tp=1, fp=1, fn=0)),
        ([], [], 1, 1, DriftConfusion(tp=0, fp=0, fn=0)),
        ([], [2, 5], 1, 1, DriftConfusion(tp=0, fp=2, fn=0)),
        ([3, 8], [], 1, 1, DriftConfusion(tp=0, fp=0, fn=2)),
        ([5], [3], 2, 3, DriftConfusion(tp=1, fp=0, fn=0)),
        ([5], [9], 2, 3, DriftConfusion(tp=0, fp=1, fn=1)),
        ([3, 10], [3, 10], 1, 1, DriftConfusion(tp=2, fp=0, fn=0)),
        ([3, 10], [3, 7, 10], 1, 1, DriftConfusion(tp=2, fp=1, fn=0)),
        ([4, 6], [5], 2, 2, DriftConfusion(tp=1, fp=0, fn=1)),
        ([0, 1], [1], 0, 1, DriftConfusion(tp=1, fp=0, fn=1)),
    ],
    ids=[
        "original_example_all_fp",
        "original_example_one_tp_one_fp",
        "empty_trues_empty_preds",
        "empty_trues_nonempty_preds",
        "nonempty_trues_empty_preds",
        "boundary_left_edge_in_window",
        "boundary_outside_window",
        "multiple_exact_matches",
        "multiple_points_one_extra_alarm",
        "overlap_tie_breaker_to_earlier_true",
        "midpoint_partition_edge_case",
    ],
)
def test_drift_confusion_examples(
    trues, preds, max_early, max_delay, expected
):
    result = drift_confusion(
        trues,
        preds,
        max_early,
        max_delay,
    )
    assert result == expected