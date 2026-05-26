import pytest
from typing import Type
from src import learner
import inspect
from capymoa.stream import Schema
from capymoa.ann import Perceptron
from src.config import converter

# Filter to get only classes, then check if they are subclasses of LearnerArgs
LEARNER_TYPES = [
    obj
    for _, obj in inspect.getmembers(learner)
    if inspect.isclass(obj) # Check if it's a class
    and issubclass(obj, learner.LearnerArgs) # Check if it's a subclass of LearnerArgs
    and obj is not learner.LearnerArgs # Exclude the base class itself
]

@pytest.mark.parametrize("learner_type", LEARNER_TYPES)
def test_learner_construction(learner_type: Type[learner.LearnerArgs]):
    schema = Schema.from_custom(
        features=["f1", "f2", "class"],
        target="class",
        categories={"class": ["A", "B"]},
    )
    model = Perceptron(schema, 1)
    config = learner_type()
    
    # Check that it can be converted to and from a dictionary
    assert converter.structure(converter.unstructure(config), learner_type) == config
    
    config.build(seed=0, schema=schema, device="cpu", model=model)
