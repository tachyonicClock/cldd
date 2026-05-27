from src.tblogger import TensorboardLogger
from src import config
from capymoa.base.events import Dispatcher
from capymoa.ocl.evaluation import OCLMetrics, ocl_train_eval_loop
from typing import Sequence
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from loguru import logger
import pickle


class Experiment:
    def __init__(self, config: config.Config):
        self.config = config
        self.device = config.device

        logger.info("Build scenario.")
        self.scenario = config.scenario.build(config.seed)
        self.schema = self.scenario.schema

        logger.info("Build model.")
        self.model = config.model.build(config.seed, self.schema)

        logger.info("Build learner (strategy).")
        self.learner = config.learner.build(
            config.seed, self.schema, self.device, self.model
        )

        logger.info("Build drift detector.")
        self.drift_detector = config.drift_detector.build(config.seed, self.learner)

        self.logdir = self.new_logdir()
        self.tb_logger = TensorboardLogger(self.logdir.as_posix())

    def new_logdir(self) -> Path:
        return self.config.logdir

    def train_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_train, shuffle=False)
        return [new_loader(task) for task in self.scenario.train_tasks]

    def test_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_test, shuffle=False)
        return [new_loader(task) for task in self.scenario.test_tasks]

    def run(self) -> OCLMetrics:
        self.logdir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving config...")
        with open(self.logdir / "config.yaml", "w") as f:
            config_dict = config.converter.unstructure(self.config)
            f.write(config.OmegaConf.to_yaml(config_dict))

        dispatcher = Dispatcher()
        self.tb_logger.attach_with(dispatcher)
        self.drift_detector.attach_with(dispatcher)

        ocl_metrics = ocl_train_eval_loop(
            learner=self.learner,
            train_streams=self.train_streams(),
            test_streams=self.test_streams(),
            progress_bar=not self.config.quiet,
            dispatcher=dispatcher,
            attach_learner=False,
        )

        logger.info(f"Saving results to {self.logdir}")
        # Save Online Metrics
        ocl_metrics.ttt.write_to_file((self.logdir / "ttt").as_posix())

        # Save Continual Learning Metrics
        ocl_pickle = self.logdir / "ocl_metrics.pkl"
        with open(ocl_pickle, "wb") as f:
            ocl_metrics.ttt = None  # type: ignore
            pickle.dump(ocl_metrics, f)

        # Save Drift Detection Metrics
        dd_metrics = self.drift_detector.metrics()
        dd_pickle = self.logdir / "dd_metrics.pkl"
        with open(dd_pickle, "wb") as f:
            pickle.dump(dd_metrics, f)

        logger.info("{}", "-" * 30)
        logger.info("DRIFT DETECTION METRICS")
        logger.info("{}", "-" * 30)
        for key, value in dd_metrics.items():
            if isinstance(value, float):
                logger.info(f"{key.ljust(20)} {value:.3f}")
            elif isinstance(value, int):
                logger.info(f"{key.ljust(20)} {value}")
        logger.info("PREDS {}", dd_metrics["preds"])
        logger.info("TRUES {}", dd_metrics["trues"])

        logger.info("{}", "-" * 30)
        logger.info("OCL METRICS")
        logger.info("{}", "-" * 30)
        logger.info(f"accuracy_seen_avg  {ocl_metrics.accuracy_seen_avg:.3f}")
        logger.info(f"accuracy_all_avg   {ocl_metrics.accuracy_all_avg:.3f}")
        logger.info(f"accuracy_final     {ocl_metrics.accuracy_final:.3f}")
        return ocl_metrics
