from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal, Union

import torch
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from capymoa.base.events import Dispatcher, Handler, LogScalar
from capymoa.ocl.evaluation.events import TrainBatchPredict

LOG_EVERY = 32


class SchedulerWrapper(Handler):
    def __init__(self, scheduler: LRScheduler):
        self.scheduler = scheduler

    def on_train_batch_predict(self, event: TrainBatchPredict) -> None:
        self.scheduler.step()

        step = event.global_step
        if step % LOG_EVERY == 0:
            lr = float(self.scheduler.get_last_lr()[0])
            self.dispatcher.notify(LogScalar("scheduler.lr", lr, event.global_step))

    def attach_with(self, dispatcher: Dispatcher) -> "Handler":
        self.dispatcher = dispatcher
        dispatcher.subscribe(TrainBatchPredict, self.on_train_batch_predict)
        return self


@dataclass
class SchedulerArgs(ABC):
    type_: ClassVar[str]

    @abstractmethod
    def build(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> LRScheduler: ...


@dataclass
class OneCycleLRArgs(SchedulerArgs):
    type_: ClassVar[str] = "OneCycleLR"

    max_lr: float = 0.001
    pct_start: float = 0.3
    anneal_strategy: Literal["cos", "linear"] = "cos"
    div_factor: float = 25.0
    final_div_factor: float = 10_000.0
    three_phase: bool = False

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> SchedulerWrapper:
        if total_steps is None:
            raise ValueError("OneCycleLR requires total_steps to build the scheduler.")

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=total_steps,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            three_phase=self.three_phase,
        )
        return SchedulerWrapper(scheduler)


AnyScheduler = Union[OneCycleLRArgs]
