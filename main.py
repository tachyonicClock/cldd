import click
from pathlib import Path
from src.dd_hpearch import DDHPSearch
from src.config import get_config, Config
from src.experiment import Experiment
from src.hpsearch import HPSearch
from cattrs import transform_error, BaseValidationError
from loguru import logger
import pprint


@click.group()
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
@click.pass_context
def cli(ctx, config: Path, dotlist: list[str]):
    try:
        config_obj = get_config(Path(config), Path("config/base"), list(dotlist))
        pprint.pprint(config_obj)
    except BaseValidationError as e:
        for error in transform_error(e):
            logger.error(f"Validation: {error}")
        exit(1)
        return
    ctx.obj = config_obj


@cli.command()
@click.argument("label", type=str)
@click.pass_context
def hpsearch(ctx, label):
    config = ctx.obj
    assert isinstance(config, Config)
    config.label = label
    HPSearch(config).optimize()


@cli.command()
@click.argument("study_name", type=str)
@click.pass_context
def dd_hpsearch(ctx, study_name):
    """Hyperparameter search drift detector using a frozen error-rate stream."""
    config = ctx.obj
    assert isinstance(config, Config)
    config.label = "dd_hpsearch"
    DDHPSearch(config, study_name).optimize()


@cli.command()
@click.pass_context
def run(ctx):
    config = ctx.obj
    assert isinstance(config, Config)
    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    cli()
