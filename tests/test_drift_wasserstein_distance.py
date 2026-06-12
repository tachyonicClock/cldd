from typing import Sequence
from scipy.stats import wasserstein_distance
from ot import uot_1d
import torch
from hypothesis import given, strategies as st
import numpy as np


def drift_wd(
    trues: Sequence[float] | np.ndarray,
    preds: Sequence[float] | np.ndarray,
    fp_penalty: float = 0.1,
    fn_penalty: float = 1,
) -> float:
    assert fp_penalty >= 0, "False positive penalty must be non-negative."
    assert fn_penalty >= 0, "False negative penalty must be non-negative."
    assert len(trues)

    # Relative penalties.
    fp_penalty = float(fp_penalty) / len(trues)
    fn_penalty = float(fn_penalty) / len(trues)

    trues_ = np.asarray(trues)
    preds_ = np.asarray(preds)

    if len(preds_) == 0:
        return float(fn_penalty * len(trues_))

    assert 0 <= trues_.min() <= trues_.max() <= 1, "Trues must be in [0, 1]."
    assert 0 <= preds_.min() <= preds_.max() <= 1, "Preds must be in [0, 1]."

    shape_loss = wasserstein_distance(trues_, preds_)

    fp_loss = (fp_penalty) * max(0, len(preds_) - len(trues_))
    fn_loss = (fn_penalty) * max(0, len(trues_) - len(preds_))
    return float(shape_loss + fp_loss + fn_loss)


def drift_uot(
    trues: Sequence[float] | torch.Tensor | np.ndarray,
    preds: Sequence[float] | torch.Tensor | np.ndarray,
    false_positive_penalty: float = 1.0,
    false_negative_penalty: float = 1.0,
) -> float:
    trues_ = torch.asarray(trues, dtype=torch.float32)
    preds_ = torch.asarray(preds, dtype=torch.float32)
    assert trues_.ndim == 1 and preds_.ndim == 1, "Trues and preds must be 1D"
    assert 0.0 <= trues_.min() and trues_.max() <= 1.0, "Trues must be in [0, 1]"
    assert 0.0 <= preds_.min() and preds_.max() <= 1.0, "Preds must be in [0, 1]"
    _, _, cost = uot_1d(  # type: ignore
        trues_.view(-1, 1),
        preds_.view(-1, 1),
        u_weights=torch.ones(len(trues)) / len(trues),
        v_weights=torch.ones(len(preds)) / len(trues),
        reg_m=(false_positive_penalty, false_negative_penalty),
        p=1,
        returnCost="total",
    )
    return float(cost.item())


@st.composite
def _two_lists(draw):
    # Draw two lists of the same length
    max_length = 100
    length = draw(st.integers(min_value=1, max_value=max_length))
    element = st.floats(0, 1)
    trues = draw(st.lists(element, min_size=length, max_size=length))
    preds = draw(st.lists(element, min_size=length, max_size=length))
    return trues, preds


@given(_two_lists())
def test_balanced_case(lists):
    trues, preds = lists
    wd_scipy = wasserstein_distance(trues, preds)
    assert np.isfinite(wd_scipy)
    wd_unbalanced = drift_wd(trues, preds, 0, 0)
    assert np.isfinite(wd_unbalanced)
    assert np.isclose(wd_scipy, wd_unbalanced, atol=1e-5)


def test_simple():
    trues = np.linspace(0, 1, 6)[1:-1]

    assert np.isclose(drift_wd(trues, trues), 0.0)

    trues_perm = trues.copy()
    np.random.shuffle(trues_perm)
    assert np.isclose(drift_wd(trues, trues_perm), 0.0)

    # Few false positives should be better than many false positives.
    many_fp = drift_wd(trues, np.linspace(0, 1, 500))
    few_fp = drift_wd(trues, np.linspace(0, 1, 10))
    assert many_fp > few_fp

    # False positives near the true values should be better than false positives far
    # from the true values.
    little_off = drift_wd(trues, np.concatenate([trues, trues + 0.05]))
    far_off = drift_wd(trues, np.concatenate([trues, trues + 0.2]))
    assert little_off < far_off

    assert drift_wd([0, 1], [0], fn_penalty=0) == 0.5
