import itertools

from src.drift_detector import ScheduleArgs, ScheduleDriftDetector


def alarms(**kwargs) -> list[int]:
    args = ScheduleArgs(**kwargs)
    args.set_stream([100] * 5, seed=0)
    return args.alarms()


def test_oracle_matches_boundaries():
    assert alarms(mode="oracle") == [100, 200, 300, 400]


def test_oracle_faults():
    assert alarms(mode="oracle", offset=-10) == [90, 190, 290, 390]
    assert len(alarms(mode="oracle", n_miss=3)) == 1
    extra = alarms(mode="oracle", n_extra=5)
    assert {100, 200, 300, 400} <= set(extra) and len(extra) <= 9


def test_periodic_and_random():
    periodic = alarms(mode="periodic", period=50)
    assert all(b - a == 50 for a, b in itertools.pairwise(periodic))
    assert len(alarms(mode="random", n_random=7)) <= 7


def test_detector_fires_on_schedule_and_survives_reset():
    d = ScheduleDriftDetector([2, 4])
    fired = []
    for _ in range(5):
        d.add_element(0.0)
        fired.append(d.detected_change())
        d.reset()
    assert fired == [False, True, False, True, False]
