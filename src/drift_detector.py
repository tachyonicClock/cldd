from typing import Any, ClassVar, Literal
from capymoa.base.events import Dispatcher, Handler, LogScalar
from capymoa.ocl.evaluation.events import (
    TrainBatchPredict,
    TrainTaskBegin,
    TrainTaskEnd
)
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import ADWIN, PageHinkley, DDM, CUSUM
from capymoa.drift.eval_detector import EvaluateDriftDetector
from dataclasses import dataclass, asdict
from loguru import logger
import numpy as np
from torch.nn.functional import cross_entropy
import torch
import enum


class ErrorStreamType(enum.Enum):
    CE = "CE"
    """A stream of cross-entropy losses for each sample, used for drift detection. The
    drift detector will be applied on this stream to detect drifts."""
    ERROR = "ERROR"
    """A bit stream where 1 means the model was correct on that sample, and 0 means it
    was incorrect."""


class OCLDD(Handler):
    def __init__(
        self,
        drift_detector: BaseDriftDetector,
        eval_dd: EvaluateDriftDetector,
        learner: Any,
        use_batch_mean: bool,
        reset_on_drift: bool,
        error_stream_type: ErrorStreamType,
    ):
        self._downstream = Dispatcher()
        """Dispatcher used to notify the learners of detected drifts."""
        if isinstance(learner, Handler):
            logger.info("Drift detector will notify the learner of detected drifts.")
            learner.attach_with(self._downstream)

        self.reset_on_drift = reset_on_drift
        self.drift_detector = drift_detector
        self.use_batch_mean = use_batch_mean
        self.error_stream_type = error_stream_type
        self._eval_dd = eval_dd
        self._ptr = 0
        self._dd_trues = []  # True positions of drifts in the scenario
        self._dd_preds = []  # Predicted positions of drifts by the drift detector
        self._ce_stream = []  # Per-instance cross-entropy loss stream for evaluation
        self._error_stream = []  # Per-instance correctness stream for evaluation
        self._n_drifts = 0

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        self.upstream = dispatcher
        dispatcher.subscribe(TrainBatchPredict, self.on_train_batch_predict)
        dispatcher.subscribe(TrainTaskBegin, self.on_train_task_begin)

        # If the drift detector is a handler itself. e.g. The oracle detector, then we
        # also attach it to the same dispatcher.
        if isinstance(self.drift_detector, Handler):
            self.drift_detector.attach_with(dispatcher)
        return self

    def log_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.upstream.notify(
            LogScalar(
                tag=tag,
                scalar_value=scalar_value,
                global_step=global_step,
            )
        )

    @torch.no_grad()
    def on_train_batch_predict(self, event: TrainBatchPredict):
        batch_size = len(event.y)
        # Compute the cross-entropy loss and error for each instance in the batch, and
        # add them to the respective streams to save for later.
        ce_losses = (
            cross_entropy(event.y_logits, event.y, reduction="none").cpu().numpy()
        )
        ce_loss = ce_losses.mean()
        errors = (event.y != event.y_pred).bool().cpu().numpy()
        error_rate = errors.mean()
        self._error_stream.append(errors)
        self._ce_stream.append(ce_losses)

        # Depending on the configuration, we either add the batch mean or each
        # individual instance to the drift detector.
        if self.use_batch_mean:
            self._ptr += batch_size
            self._add_element(ce_loss, error_rate)
            self._poll_drift(event)
        else:
            for i in range(batch_size):
                self._ptr += 1
                self._add_element(ce_losses[i], errors[i])
                self._poll_drift(event)

        # Log the error rate and cross-entropy
        self.log_scalar("dd/error_rate", error_rate, event.global_step)
        self.log_scalar("dd/ce_loss", ce_loss, event.global_step)

    def _add_element(self, ce_loss, error_rate):
        if self.error_stream_type == ErrorStreamType.ERROR:
            self.drift_detector.add_element(error_rate)
        elif self.error_stream_type == ErrorStreamType.CE:
            self.drift_detector.add_element(ce_loss)
        else:
            raise ValueError(f"Unsupported error stream type: {self.error_stream_type}")

    def _poll_drift(self, event: TrainBatchPredict):
        if self.drift_detector.detected_change():
            logger.info(f"Predicted drift at instance {self._ptr}")
            self._dd_preds.append(self._ptr)
            self._n_drifts += 1
            self._downstream.notify(
                TrainTaskBegin(
                    train_task=self._n_drifts,
                    global_step=event.global_step,
                    train_step=event.train_step,
                )
            )
            if self.reset_on_drift:
                self.drift_detector.reset()

        if self.drift_detector.detected_warning():
            logger.warning(f"Predicted warning at instance {self._ptr}")

    def on_train_task_begin(self, event: TrainTaskBegin):
        # The start does not count as a drift.
        if event.train_task == 0:
            return
        self._dd_trues.append(self._ptr)
        logger.info("True drift at instance {}".format(self._ptr))

    def metrics(self) -> dict:
        metrics = asdict(
            self._eval_dd.calc_performance(self._dd_trues, self._dd_preds, self._ptr)
        )
        metrics["trues"] = self._dd_trues
        metrics["preds"] = self._dd_preds
        metrics["tot_n_instances"] = self._ptr

        # Save the error streams as well for further analysis.
        metrics["error_stream"] = np.concatenate(self._error_stream).astype(np.bool_)
        metrics["ce_stream"] = np.concatenate(self._ce_stream).astype(np.float16)
        return metrics

class OracleDriftDetector(BaseDriftDetector, Handler):
    def __init__(self):
        super().__init__()
        self._in_concept_change = False

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        dispatcher.subscribe(TrainTaskBegin, self.on_train_task_begin)
        return self
    
    def on_train_task_begin(self, event: TrainTaskBegin):
        if event.train_task == 0:
            return
        self._in_concept_change = True

    def add_element(self, element: float) -> None:
        pass

    def detected_change(self) -> bool:
        if self._in_concept_change:
            self._in_concept_change = False
            return True
        return False

    def get_params(self) -> dict:
        return {}

@dataclass
class DriftDetectorArgs:
    type_: ClassVar[str]

    max_delay: int = int(1000 / 4)
    rate_period: int = 1000
    max_early_detection: int = 0

    reset_on_drift: bool = True
    use_batch_mean: bool = False
    error_stream_type: Literal["CE", "ERROR"] = "CE"

    def build_eval_dd(self) -> EvaluateDriftDetector:
        return EvaluateDriftDetector(
            max_delay=self.max_delay,
            rate_period=self.rate_period,
            max_early_detection=self.max_early_detection,
        )

    def _build_dd(self, seed: int, learner: Any) -> BaseDriftDetector:
        raise NotImplementedError(
            "Must implement _build method for drift detector config"
        )

    def build(self, seed: int, learner: Any) -> "OCLDD":
        return OCLDD(
            drift_detector=self._build_dd(),
            eval_dd=self.build_eval_dd(),
            learner=learner,
            use_batch_mean=self.use_batch_mean,
            reset_on_drift=self.reset_on_drift,
            error_stream_type=ErrorStreamType(self.error_stream_type),
        )


@dataclass
class ADWINArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "ADWIN"
    delta: float = 0.002

    def _build_dd(self) -> BaseDriftDetector:
        return ADWIN(self.delta)


@dataclass
class CUSUMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "CUSUM"
    min_n_instances: int = 30
    """The minimum number of instances before permitting detecting change."""
    delta: float = 0.005
    """Delta parameter of the CUSUM test."""
    lambda_: float = 50
    """Threshold parameter of the CUSUM test."""

    def _build_dd(self) -> BaseDriftDetector:
        return CUSUM(
            min_n_instances=self.min_n_instances,
            delta=self.delta,
            lambda_=self.lambda_,
        )


@dataclass
class DDMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "DDM"
    min_n_instances: int = 30
    warning_level: float = 2.0
    out_control_level: float = 3.0

    def _build_dd(self) -> BaseDriftDetector:
        return DDM(
            min_n_instances=self.min_n_instances,
            warning_level=self.warning_level,
            out_control_level=self.out_control_level,
        )

@dataclass
class PageHinkleyArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "PageHinkley"
    min_n_instances: int = 30
    delta: float = 0.005
    lambda_: float = 50.0
    alpha: float = 0.9999

    def _build_dd(self) -> BaseDriftDetector:
        return PageHinkley(
            min_n_instances=self.min_n_instances,
            delta=self.delta,
            lambda_=self.lambda_,
            alpha=self.alpha,
        )

@dataclass
class OracleArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "oracle"

    def _build_dd(self) -> BaseDriftDetector:
        return OracleDriftDetector()


AnyDriftDetector = ADWINArgs | CUSUMArgs | DDMArgs | OracleArgs
