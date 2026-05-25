from src.tblogger import TensorboardLogger
from src import config
from capymoa.base.events import Dispatcher
from capymoa.ocl.evaluation import ocl_train_eval_loop
from typing import Sequence
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
import time
from loguru import logger
import pickle


class Experiment:
    def __init__(self, config: config.Config):
        self.config = config
        self.device = config.device

        logger.info("Build scenario.")
        self.scenario = config.scenario.build(config.seed)
        self.schema = self.scenario.schema

        logger.info("Build drift detector.")
        # self.drift_detector = config.drift_detector.build(config.seed)

        logger.info("Build model.")
        self.model = config.model.build(config.seed, self.schema)

        logger.info("Build learner (strategy).")
        self.learner = config.learner.build(
            config.seed, self.schema, self.device, self.model
        )

        self.logdir = self.new_logdir()
        self.tb_logger = TensorboardLogger(self.logdir.as_posix())

    def new_logdir(self) -> Path:
        if self.config.trial is None:
            trial_id = time.strftime("%Y%m%d-%H%M%S")
        else:
            trial_id = f"{self.config.trial:03d}"

        return (
            Path("logs")
            / (self.config.name or "unnamed")
            / f"{self.config.scenario.name}_{self.config.scenario.gradual}"
            / f"{self.config.learner.type_}_{self.config.drift_detector.type_}_{self.config.model.type_}"
            / trial_id
        )

    def train_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_train, shuffle=False)
        return [new_loader(task) for task in self.scenario.train_tasks]

    def test_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_eval, shuffle=False)
        return [new_loader(task) for task in self.scenario.test_tasks]

    def run(self):
        self.logdir.mkdir(parents=True, exist_ok=True)

        logger.info("Saving config...")
        with open(self.logdir / "config.yaml", "w") as f:
            config_dict = config.converter.unstructure(self.config)
            f.write(config.OmegaConf.to_yaml(config_dict))

        logger.info("Attach handlers to event dispatcher")
        dispatcher = Dispatcher()
        self.tb_logger.attach_with(dispatcher)

        results = ocl_train_eval_loop(
            learner=self.learner,
            train_streams=self.train_streams(),
            test_streams=self.test_streams(),
            progress_bar=True,
            dispatcher=dispatcher,
            attach_learner=False,
        )

        logger.info(f"Saving results to {self.logdir}")
        results.ttt.write_to_file((self.logdir / "ttt").as_posix())
        results_pickle = self.logdir / "ocl_metrics.pkl"
        with open(results_pickle, "wb") as f:
            results.ttt = None  # type: ignore
            pickle.dump(results, f)

        logger.info(f"accuracy_seen_avg  {results.accuracy_seen_avg:.3f}")
        logger.info(f"accuracy_all_avg   {results.accuracy_all_avg:.3f}")
        logger.info(f"accuracy_final     {results.accuracy_final:.3f}")

        return results
