from dataclasses import dataclass

from capymoa.ocl.events import Event


@dataclass
class LogScalar(Event):
    tag: str
    scalar_value: float
    global_step: int
