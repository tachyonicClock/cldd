from src.scenario import ScenarioArgs
import pytest


@pytest.mark.parametrize(
    "name,normalize_features,gradual,gradual_width",
    [
        ("DomainCIFAR100", True, True, 0.0),
        ("DomainCIFAR100", True, True, 0.5),
        ("DomainCIFAR100", True, True, 1.0),
        ("RotatedMNIST", False, False, None),
        ("RotatedFashionMNIST", False, False, None),
    ],
)
def test_scenario(name, normalize_features, gradual, gradual_width):
    config = ScenarioArgs(name, normalize_features, gradual, gradual_width)
    scenario = config.build(0)
    assert scenario.train_tasks
    assert scenario.test_tasks
    assert scenario.valid_tasks is None
