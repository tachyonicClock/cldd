from dataclasses import asdict

from src.tblogger import TensorboardLogger
from src import config
from capymoa.base.events import Dispatcher
from capymoa.drift.eval_detector import EvaluateDriftDetector
from capymoa.ocl.evaluation import ocl_train_eval_loop
from typing import Dict, Sequence
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from loguru import logger
import pickle
from shutil import rmtree
import numpy as np


class Experiment:
    def __init__(self, config: config.Config):
        self.config = config
        self.device = config.device

        logger.info("Build scenario.")
        self.scenario = config.scenario.build(config.seed)
        self.schema = self.scenario.schema

        logger.info("Build model.")
        self.model = config.model.build(config.seed, self.schema)
        n_parameters = sum(p.numel() for p in self.model.parameters())
        n_buffers = sum(p.numel() for p in self.model.buffers())
        logger.info(f"Model has {n_parameters} parameters and {n_buffers} buffers.")

        logger.info("Build learner (optimizer).")
        self.optimizer = config.optimizer.build_optimizer(self.model.parameters())

        self.scheduler = None
        if config.scheduler is not None:
            logger.info("Build learner (scheduler).")
            total = sum(len(task) for task in self.scenario.train_tasks)  # type: ignore
            self.scheduler = config.scheduler.build(
                self.optimizer,
                total_steps=int(total / config.mb_train + 100),  # type: ignore
            )

        logger.info("Build learner (strategy).")
        self.learner = config.learner.build(
            config.seed, self.schema, self.device, self.model, self.optimizer
        )

        logger.info("Build drift detector.")
        self._init_drift_detector(config)

        self.logdir = self.new_logdir()
        self.tb_logger = TensorboardLogger(self.logdir.as_posix())

    def _init_drift_detector(self, config):
        task_lengths = np.array([len(task) for task in self.scenario.train_tasks])  # type: ignore
        mean_length = task_lengths.mean()
        pad = self.config.mb_train * 10
        drift_width = int(mean_length * self.config.scenario.gradual + pad)
        self.drift_detector = config.drift_detector.build(
            self.learner,
            EvaluateDriftDetector(
                max_delay=drift_width,
                max_early_detection=drift_width,
                rate_period=int(mean_length),
            ),
        )

    def new_logdir(self) -> Path:
        logdir = self.config.logdir
        if logdir.exists():
            logger.warning("Replacing pre-existing logdir at {}", logdir)
            rmtree(logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        return logdir

    def train_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_train, shuffle=False)
        return [new_loader(task) for task in self.scenario.train_tasks]

    def test_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_test, shuffle=False)
        return [new_loader(task) for task in self.scenario.test_tasks]

    def run(self) -> Dict:
        self.logdir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving config...")
        with open(self.logdir / "config.yaml", "w") as f:
            config_dict = config.converter.unstructure(self.config)
            f.write(config.OmegaConf.to_yaml(config_dict))

        self.dispatcher = Dispatcher()
        self.tb_logger.attach_with(self.dispatcher)
        self.drift_detector.attach_with(self.dispatcher)
        if self.scheduler is not None:
            self.scheduler.attach_with(self.dispatcher)

        ocl_metrics = ocl_train_eval_loop(
            learner=self.learner,
            train_streams=self.train_streams(),
            test_streams=self.test_streams(),
            progress_bar=not self.config.disable_progress_bar,
            dispatcher=self.dispatcher,
            attach_learner=False,
            epochs=self.config.scenario.epochs,
        )

        logger.info(f"Saving results to {self.logdir}")
        ttt_pickle = self.logdir / "ttt_metrics.pkl"
        with open(ttt_pickle, "wb") as f:
            pickle.dump(
                {
                    "cumulative": ocl_metrics.ttt.cumulative.metrics_dict(),  # type: ignore
                    "windowed": ocl_metrics.ttt.metrics_per_window(),
                },
                f,
            )  # type: ignore

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
        logger.info("METRICS")
        logger.info("{}", "-" * 30)
        for key, value in (dd_metrics | asdict(ocl_metrics)).items():
            if isinstance(value, float):
                logger.info(f"{key.ljust(20)} {value:.3f}")
            elif isinstance(value, int):
                logger.info(f"{key.ljust(20)} {value}")
        logger.info("PREDS {}", dd_metrics["preds"])
        logger.info("TRUES {}", dd_metrics["trues"])

        accuracy_forgetful = ocl_metrics.accuracy_matrix.trace() / ocl_metrics.n_tasks
        logger.info(f"accuracy_forgetful {accuracy_forgetful:.3f}")
        logger.info(f"accuracy_seen_avg  {ocl_metrics.accuracy_seen_avg:.3f}")
        logger.info(f"accuracy_all_avg   {ocl_metrics.accuracy_all_avg:.3f}")
        logger.info(f"accuracy_final     {ocl_metrics.accuracy_final:.3f}")
        ocl_metrics = asdict(ocl_metrics)
        ocl_metrics["accuracy_forgetful"] = accuracy_forgetful
        return ocl_metrics
