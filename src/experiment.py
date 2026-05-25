from src import config
from capymoa.ocl.evaluation import ocl_train_eval_loop
from typing import Sequence
from torch.utils.data import DataLoader
from functools import partial


class Experiment:
    def __init__(self, config: config.Config):
        self.config = config
        self.scenario = config.scenario.build(config.seed)
        self.drift_detector = config.drift_detector.build(config.seed)
        self.model = config.model.build(
            config.seed, self.scenario.schema.get_num_classes()
        )
        self.learner = config.learner.build(config.seed, self.model)

    def train_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_train, shuffle=False)
        return [new_loader(task) for task in self.scenario.train_tasks]

    def test_streams(self) -> Sequence[DataLoader]:
        new_loader = partial(DataLoader, batch_size=self.config.mb_eval, shuffle=False)
        return [new_loader(task) for task in self.scenario.test_tasks]

    def run(self):
        results = ocl_train_eval_loop(
            learner=self.learner,
            train_streams=self.train_streams(),
            test_streams=self.test_streams(),
        )
        return results
