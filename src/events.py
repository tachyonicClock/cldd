from capymoa.base.events import Event
from dataclasses import dataclass


@dataclass
class LogScalar(Event):
    tag: str
    scalar_value: float
    global_step: int
