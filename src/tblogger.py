from torch.utils.tensorboard import SummaryWriter
from capymoa.ocl.events import Dispatcher, Handler

from src.events import LogScalar
from capymoa.ocl.evaluation.events import TrainEnd


class TensorboardLogger(Handler):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, event: LogScalar):
        self.writer.add_scalar(event.tag, event.scalar_value, event.global_step)

    def on_train_end(self, event: TrainEnd):
        self.writer.flush()
        self.writer.close()

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        dispatcher.subscribe(LogScalar, self.log_scalar)
        dispatcher.subscribe(TrainEnd, self.on_train_end)
        return self
