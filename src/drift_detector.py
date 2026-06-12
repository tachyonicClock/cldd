from typing import Any, ClassVar, Literal, Sequence
from capymoa.base.events import Dispatcher, Handler, LogScalar
from capymoa.ocl.evaluation.events import (
    TrainBatchPredict,
    TrainTaskBegin,
    TrainTaskEnd,
    TestBegin,
    TestEnd,
)
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.drift.detectors import (
    ABCD,
    ADWIN,
    DDM,
    PageHinkley,
    SEED,
    STEPD,
)
from capymoa.drift.eval_detector import EvaluateDriftDetector
from dataclasses import dataclass, asdict
from loguru import logger
import numpy as np
from torch.nn.functional import cross_entropy
import torch
import enum
from scipy.stats import wasserstein_distance

LOG_EVERY = 10


def drift_wd(
    trues: Sequence[float] | np.ndarray,
    preds: Sequence[float] | np.ndarray,
    fp_penalty: float = 0,
    fn_penalty: float = 0,
) -> float:
    assert fp_penalty >= 0, "False positive penalty must be non-negative."
    assert fn_penalty >= 0, "False negative penalty must be non-negative."
    assert len(trues)

    # Relative penalties.
    fp_penalty = float(fp_penalty) / len(trues)
    fn_penalty = float(fn_penalty) / len(trues)

    trues_ = np.asarray(trues)
    preds_ = np.asarray(preds)

    if len(preds_) == 0:
        return float(fn_penalty * len(trues_))

    assert 0 <= trues_.min() <= trues_.max() <= 1, "Trues must be in [0, 1]."
    assert 0 <= preds_.min() <= preds_.max() <= 1, "Preds must be in [0, 1]."

    shape_loss = wasserstein_distance(trues_, preds_)

    fp_loss = (fp_penalty) * max(0, len(preds_) - len(trues_))
    fn_loss = (fn_penalty) * max(0, len(trues_) - len(preds_))
    return float(shape_loss + fp_loss + fn_loss)


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
        self.error_stream_type = error_stream_type
        self._eval_dd = eval_dd
        self._stream_index = 0
        self._dd_trues = []  # True positions of drifts in the scenario
        self._dd_preds = []  # Predicted positions of drifts by the drift detector
        self._ce_stream = []  # Per-instance cross-entropy loss stream for evaluation
        self._error_stream = []  # Per-instance correctness stream for evaluation
        self._n_drifts = 0

    def attach_with(self, dispatcher: Dispatcher) -> Handler:
        self.upstream = dispatcher
        dispatcher.subscribe(TrainBatchPredict, self.on_train_batch_predict)
        dispatcher.subscribe(TrainTaskBegin, self.on_train_task_begin)
        dispatcher.subscribe(TestBegin, self._proxy)
        dispatcher.subscribe(TestEnd, self._proxy)

        # If the drift detector is a handler itself. e.g. The oracle detector, then we
        # also attach it to the same dispatcher.
        if isinstance(self.drift_detector, Handler):
            self.drift_detector.attach_with(dispatcher)
        return self

    def log_scalar(self, tag: str, scalar_value: float, global_step: int):
        if global_step % LOG_EVERY == 0:
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

        # Add each individual instance to the drift detector.
        for i in range(batch_size):
            self._stream_index += 1
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
            logger.info(f"Predicted drift at instance {self._stream_index}")
            self._dd_preds.append(self._stream_index)

            self._downstream.notify(
                TrainTaskEnd(
                    train_task=self._n_drifts,
                    global_step=event.global_step,
                    train_step=event.train_step,
                )
            )
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
            logger.warning(f"Predicted warning at instance {self._stream_index}")

    def on_train_task_begin(self, event: TrainTaskBegin):
        # The start does not count as a drift.
        if event.train_task == 0:
            # We always know about the first task.
            self._downstream.notify(event)
            return
        self._dd_trues.append(self._stream_index)
        logger.info("True drift at instance {}".format(self._stream_index))

    def _proxy(self, event):
        self._downstream.notify(event)

    def metrics(self) -> dict:
        metrics = asdict(
            self._eval_dd.calc_performance(
                self._dd_trues, self._dd_preds, self._stream_index
            )
        )
        trues_rel = np.array(self._dd_trues) / self._stream_index
        preds_rel = np.array(self._dd_preds) / self._stream_index
        metrics["trues"] = self._dd_trues
        metrics["preds"] = self._dd_preds
        metrics["wasserstein_distance"] = drift_wd(trues_rel, preds_rel)
        metrics["tot_n_instances"] = self._stream_index
        metrics["max_delay"] = self._eval_dd.max_delay
        metrics["max_early_detection"] = self._eval_dd.max_early_detection
        metrics["rate_period"] = self._eval_dd.rate_period

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

    # max_delay: int = 500
    # rate_period: int = 1000
    # max_early_detection: int = 0

    reset_on_drift: bool = True
    error_stream_type: Literal["CE", "ERROR"] = "CE"

    label: str | None = None

    def build_dd(self) -> BaseDriftDetector:
        raise NotImplementedError(
            "Must implement _build method for drift detector config"
        )

    def build(self, learner: Any, evaluator: EvaluateDriftDetector) -> "OCLDD":
        return OCLDD(
            drift_detector=self.build_dd(),
            eval_dd=evaluator,
            learner=learner,
            reset_on_drift=self.reset_on_drift,
            error_stream_type=ErrorStreamType(self.error_stream_type),
        )


@dataclass
class ADWINArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "ADWIN"
    delta: float = 0.002

    def build_dd(self) -> BaseDriftDetector:
        return ADWIN(self.delta)


@dataclass
class DDMArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "DDM"
    min_n_instances: int = 30
    warning_level: float = 2.0
    out_control_level: float = 3.0

    def build_dd(self) -> BaseDriftDetector:
        return DDM(
            min_n_instances=self.min_n_instances,
            warning_level=self.warning_level,
            out_control_level=self.out_control_level,
        )


@dataclass
class PHArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "PH"
    min_n_instances: int = 30
    delta: float = 0.005
    lambda_: float = 50.0
    alpha: float = 0.9999

    def build_dd(self) -> BaseDriftDetector:
        return PageHinkley(
            min_n_instances=self.min_n_instances,
            delta=self.delta,
            lambda_=self.lambda_,
            alpha=self.alpha,
        )


@dataclass
class SEEDArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "SEED"
    delta: float = 0.05
    block_size: int = 32
    epsilon_prime: float = 0.01
    alpha: float = 0.8
    compress_term: int = 75

    def build_dd(self) -> BaseDriftDetector:
        return SEED(
            delta=self.delta,
            block_size=self.block_size,
            epsilon_prime=self.epsilon_prime,
            alpha=self.alpha,
            compress_term=self.compress_term,
        )


@dataclass
class STEPDArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "STEPD"
    window_size: int = 30
    alpha_drift: float = 0.003
    alpha_warning: float = 0.05

    def build_dd(self) -> BaseDriftDetector:
        return STEPD(
            window_size=self.window_size,
            alpha_drift=self.alpha_drift,
            alpha_warning=self.alpha_warning,
        )


@dataclass
class ABCDArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "ABCD"
    delta_drift: float = 0.002
    delta_warn: float = 0.01
    model_id: str = "ae"
    split_type: str = "ed"
    encoding_factor: float = 0.5
    update_epochs: int = 50
    num_splits: int = 20
    max_size: float = float("inf")
    subspace_threshold: float = 2.5
    n_min: int = 100
    maximum_absolute_value: float = 1.0
    bonferroni: bool = False

    def build_dd(self) -> BaseDriftDetector:
        return ABCD(
            delta_drift=self.delta_drift,
            delta_warn=self.delta_warn,
            model_id=self.model_id,
            split_type=self.split_type,
            encoding_factor=self.encoding_factor,
            update_epochs=self.update_epochs,
            num_splits=self.num_splits,
            max_size=self.max_size,
            subspace_threshold=self.subspace_threshold,
            n_min=self.n_min,
            maximum_absolute_value=self.maximum_absolute_value,
            bonferroni=self.bonferroni,
        )


@dataclass
class OracleArgs(DriftDetectorArgs):
    type_: ClassVar[str] = "oracle"

    def build_dd(self) -> BaseDriftDetector:
        return OracleDriftDetector()


AnyDriftDetector = (
    ADWINArgs | DDMArgs | PHArgs | SEEDArgs | STEPDArgs | ABCDArgs | OracleArgs
)
