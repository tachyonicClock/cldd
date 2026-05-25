import click
from pathlib import Path
from src.config import get_config
from src.experiment import Experiment
from cattrs import transform_error, BaseValidationError
from loguru import logger
import pprint


@click.group()
def cli(): ...


@cli.command()
def hpearch(): ...


@cli.command()
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--dotlist",
    "-a",
    multiple=True,
    help="Additional config overrides in dotlist format.",
)
def run(config: str, dotlist: list[str]):
    try:
        config_obj = get_config(Path(config), Path("config/base"), list(dotlist))
        pprint.pprint(config_obj)
        experiment = Experiment(config_obj)
        experiment.run()
    except BaseValidationError as e:
        for error in transform_error(e):
            logger.error(f"Validation: {error}")
        exit(1)
        return


if __name__ == "__main__":
    cli()
