"""The Python SEED and STEPD must match MOA when two-sided, and ignore falls in error
when one-sided."""

import numpy as np
import pytest
from capymoa.drift.detectors import SEED as MoaSEED
from capymoa.drift.detectors import STEPD as MoaSTEPD

from src.one_sided_detectors import SEED, STEPD


def piecewise_errors(seed: int, levels: list[float], length: int = 4000) -> np.ndarray:
    """Error bits whose rate steps through `levels`, `length` instances each."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.random(length) < p for p in levels]).astype(float)


def alarms(detector, stream: np.ndarray, reset: bool = True) -> list[int]:
    """Positions of the alarms. As in the experiments, reset the detector after one."""
    found = []
    for i, x in enumerate(stream):
        detector.add_element(x)
        if detector.detected_change():
            found.append(i)
            if reset:
                detector.reset()
    return found


STREAMS = [
    (0, [0.2, 0.6, 0.2, 0.7]),  # rises and falls
    (1, [0.5, 0.1, 0.4, 0.05]),  # falls first
    (2, [0.05, 0.05, 0.3, 0.3]),  # a single rise
    (3, [0.0, 0.0, 0.5]),  # long runs of zeros
    (4, [0.3, 0.3, 0.3]),  # no change
]


@pytest.mark.parametrize("seed,levels", STREAMS)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"window_size": 100, "alpha_drift": 0.01}, {"window_size": 400}],
)
def test_stepd_matches_moa_when_two_sided(seed, levels, kwargs):
    stream = piecewise_errors(seed, levels)
    expected = alarms(MoaSTEPD(**kwargs), stream)
    got = alarms(STEPD(one_sided=False, **kwargs), stream)
    assert got == expected


@pytest.mark.parametrize("seed,levels", STREAMS)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"delta": 0.002, "block_size": 64}, {"compress_term": 50, "alpha": 0.4}],
)
def test_seed_matches_moa_when_two_sided(seed, levels, kwargs):
    stream = piecewise_errors(seed, levels)
    expected = alarms(MoaSEED(**kwargs), stream)
    got = alarms(SEED(one_sided=False, **kwargs), stream)
    assert got == expected


# The error steps at instance 4000. STEPD's short window also fires on noise in a steady
# stream, so the tests look only at what happens after the step.
STEP = 4000


@pytest.mark.parametrize("detector_type", [STEPD, SEED])
def test_one_sided_ignores_a_fall_in_error(detector_type):
    stream = piecewise_errors(0, [0.6, 0.1])
    two_sided = [a for a in alarms(detector_type(one_sided=False), stream) if a >= STEP]
    one_sided = [a for a in alarms(detector_type(one_sided=True), stream) if a >= STEP]
    assert two_sided, "the two-sided detector should fire on the fall"
    assert one_sided == []


@pytest.mark.parametrize("detector_type", [STEPD, SEED])
def test_one_sided_still_fires_on_a_rise_in_error(detector_type):
    stream = piecewise_errors(0, [0.1, 0.6])
    found = [a for a in alarms(detector_type(one_sided=True), stream) if a >= STEP]
    assert found and found[0] < STEP + 800
