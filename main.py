import click
import sys
from pathlib import Path
from src.dd_hpsearch import DDHPSearch
from src.config import get_config, Config
from src.experiment import Experiment
from src.hpsearch import HPSearch
from cattrs import transform_error, BaseValidationError
from loguru import logger
from omegaconf import OmegaConf


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
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Only show warnings and errors.",
)
@click.pass_context
def cli(ctx, config: Path, dotlist: list[str], quiet: bool):
    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    try:
        config_obj = get_config(Path(config), Path("config/base"), list(dotlist))
        config_obj.quiet = quiet
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
    hpsearch_ = HPSearch(config)
    hpsearch_.optimize()
    # Save best config to 'logs/hp/00_abrupt/FT_oracle_MLP'
    best_params = hpsearch_.study.best_trial.params
    config.logdir.parent.mkdir(parents=True, exist_ok=True)
    with open(config.logdir.parent / "best_params.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(best_params))


@cli.command(name="dd_hpsearch")
@click.argument("source", type=str)
@click.pass_context
def dd_hpsearch(ctx, source):
    """Hyperparameter search drift detector using a study name or metrics pickle."""
    config = ctx.obj
    assert isinstance(config, Config)
    config.label = "dd_hpsearch"
    DDHPSearch(config, source).optimize()


@cli.command()
@click.pass_context
def run(ctx):
    config = ctx.obj
    assert isinstance(config, Config)
    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    cli()
