import click
from pathlib import Path
from src.config import get_config
from cattrs import transform_error, BaseValidationError
from loguru import logger


@click.group()
def cli(): ...


@cli.command()
def hpearch(): ...


@cli.command()
@click.option(
    "--config",
    "-c",
    required=True,
    help="Path to the config file.",
    type=click.Path(exists=True),
)
@click.option(
    "--dotlist",
    "-a",
    multiple=True,
    help="Additional config overrides in dotlist format.",
)
def run(config: str, dotlist: list[str]):
    try:
        get_config(Path(config), Path("config/base"), list(dotlist))
    except BaseValidationError as e:
        for error in transform_error(e):
            logger.error(f"Validation: {error}")
        exit(1)
        return


if __name__ == "__main__":
    cli()
