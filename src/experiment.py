from src import config
from dataclasses import dataclass


@dataclass
class Result:
    config: config.Config





def execute_experiment(config: Config) -> Result:
    scenario = get_scenario(config)

    return Result(
        config=config,
    )

