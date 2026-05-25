from typing import Any, ClassVar
from capymoa.base.events import Dispatcher, Handler, LogScalar
from capymoa.ocl.evaluation.events import TrainBatchPredict, TrainTaskEnd
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN
from capymoa.drift.eval_detector import EvaluateDriftDetector
from dataclasses import dataclass, asdict
from loguru import logger
import numpy as np


class OCLDD(Handler):
    def __init__(
        self,
        drift_detector: BaseDriftDetector,
        eval_dd: EvaluateDriftDetector,
        learner: Any,
    ):
        self.drift_detector = drift_detector
        self._downstream = Dispatcher()
        if isinstance(learner, Handler):
            logger.info("Drift detector will notify the learner of detected drifts.")
            learner.attach_with(self._downstream)
        """Dispatcher used to notify the learners of detected drifts."""
        self._eval_dd = eval_dd
        self._ptr = 0
        self._dd_trues = []  # True positions of drifts in the scenario
        self._dd_preds = []  # Predicted positions of drifts by the drift detector
        self._dd_corrects = []  # Was the model correct on each sample, used for evaluation of drift detector offline

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        self.upstream = dispatcher
        dispatcher.subscribe(TrainBatchPredict, self.on_train_batch_predict)
        dispatcher.subscribe(TrainTaskEnd, self.on_train_task_end)

        # If the drift detector is a handler itself. e.g. The oracle detector, then we
        # also attach it to the same dispatcher.
        if isinstance(self.drift_detector, Handler):
            self.drift_detector.attach_with(dispatcher)
        return self

    def on_train_batch_predict(self, event: TrainBatchPredict):
        self._ptr += event.y.size(0)
        correct = (event.y == event.y_pred).bool().numpy()
        self._dd_corrects.append(correct)

        for c in correct:
            self.drift_detector.add_element(c)

        if self.drift_detector.detected_change():
            logger.info(f"Detected drift at {event.global_step} (global_step).")
            self._dd_preds.append(self._ptr)

        self.upstream.notify(
            LogScalar(
                tag="drift_detector_stream",
                scalar_value=correct.sum() / correct.size,
                global_step=event.global_step,
            )
        )

    def on_train_task_end(self, event: TrainTaskEnd):
        logger.info(
            f"End of task {event.train_task} at {event.global_step} (global_step)."
        )
        self._dd_trues.append(self._ptr)

    def metrics(self) -> dict:
        metrics = asdict(
            self._eval_dd.calc_performance(self._dd_trues, self._dd_preds, self._ptr)
        )
        metrics["trues"] = self._dd_trues
        metrics["preds"] = self._dd_preds
        metrics["tot_n_instances"] = self._ptr
        metrics["corrects"] = np.concatenate(self._dd_corrects)
        return metrics


@dataclass
class DriftDetectorArgs:
    type_: ClassVar[str]

    max_delay: int = 100
    rate_period: int = 1000
    max_early_detection: int = 0

    def build_eval_dd(self) -> EvaluateDriftDetector:
        return EvaluateDriftDetector(
            max_delay=self.max_delay,
            rate_period=self.rate_period,
            max_early_detection=self.max_early_detection,
        )

    def build(self, seed: int, learner: Any) -> "OCLDD":
        raise NotImplementedError(
            "Must implement build method for drift detector config"
        )


@dataclass
class ADWINArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "ADWIN"
    delta: float = 0.002

    def build(self, seed: int, learner: Any) -> OCLDD:

        dd = ADWIN(self.delta)
        logger.info(dd.CLI)
        return OCLDD(dd, self.build_eval_dd(), learner)


@dataclass
class EDDMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "EDDM"


@dataclass
class DDMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "DDM"


@dataclass
class OracleArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "oracle"


AnyDriftDetector = ADWINArgs | EDDMArgs | DDMArgs | OracleArgs
