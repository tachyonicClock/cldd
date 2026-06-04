import click
from typing import Any, Dict
from pathlib import Path
import pickle

import optuna
from src.dd_hpsearch import DDHPSearch
from src.dd_eval import evaluate_dd_stream
from src.config import get_config, Config
from src.experiment import Experiment
from src.hpsearch import HPSearch
from cattrs import transform_error, BaseValidationError
from loguru import logger
from omegaconf import OmegaConf
import sys

file = Path(__file__).resolve()


def _log_format(record):
    path = Path(record["file"].path).relative_to(file.parent)
    record["path_no"] = f"{path}:{record['line']}"
    return "<cyan>{path_no: >30}</cyan> <level>{message}</level>\n"


logger.remove()
logger.add(sys.stderr, format=_log_format)
logger.info("Starting experiment with config files:")


def dotlist_dict_to_nested(dotlist_dict: Dict[str, Any]) -> dict:
    dotlist = [f"{k}={v}" for k, v in dotlist_dict.items()]
    return dict(OmegaConf.from_dotlist(dotlist))


def trial_to_yml(directory: Path, config: Config, trial: optuna.trial.FrozenTrial):
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "best_params.yml"
    yaml = OmegaConf.to_yaml(dotlist_dict_to_nested(trial.params))
    with open(file, "w") as f:
        f.write(f"# Scored {trial.value}\n")
        f.write(yaml)

    trial_dict = {
        "number": trial.number,
        "value": trial.value,
        "params": dotlist_dict_to_nested(trial.params),
        "user_attrs": trial.user_attrs,
        "system_attrs": trial.system_attrs,
        "state": trial.state.name,
        "config": config.apply_dotlist_dict(trial.params).dump(),
    }
    yaml = OmegaConf.to_yaml(trial_dict)
    with open(directory / "best_trial.yml", "w") as f:
        f.write(yaml)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    multiple=True,
    help="Path to config file(s). Can specify multiple, in which case they will be merged with later files taking precedence.",
)
@click.option(
    "--dotlist",
    "-a",
    multiple=True,
    help="Additional config overrides in dotlist format.",
)
@click.pass_context
def cli(ctx, config: list[str], dotlist: list[str]):
    try:
        config_obj = get_config(config, dotlist)
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

    # Save best config as YAML for reference.
    trial_to_yml(config.logdir.parent, config, hpsearch_.study.best_trial)


@cli.command(name="dd_hpsearch")
@click.argument(
    "error_streams",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_context
def dd_hpsearch(ctx, error_streams):
    """Hyperparameter search drift detector using a study name or metrics pickle."""
    config = ctx.obj
    assert isinstance(config, Config)
    hpsearch_ = DDHPSearch(config, error_streams)
    hpsearch_.optimize()

    # Save best config as YAML for reference.
    trial_to_yml(config.logdir.parent, config, hpsearch_.study.best_trial)


@cli.command(name="dd_run")
@click.argument(
    "error_stream",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.pass_context
def dd_run(ctx, error_stream):
    """Run drift detector evaluation on a single frozen error-rate stream."""
    config = ctx.obj
    assert isinstance(config, Config)

    with open(error_stream, "rb") as f:
        dd_metrics = pickle.load(f)

    metrics = evaluate_dd_stream(config.drift_detector, dd_metrics)
    config.logdir.mkdir(parents=True, exist_ok=True)
    with open(config.logdir / "dd_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    logger.info("Saved drift-detector metrics to {}", config.logdir / "dd_metrics.pkl")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key.ljust(20)} {value:.3f}")
        elif isinstance(value, int):
            logger.info(f"{key.ljust(20)} {value}")
    logger.info("PREDS {}", metrics["preds"])
    logger.info("TRUES {}", metrics["trues"])


@cli.command()
@click.pass_context
def run(ctx):
    config = ctx.obj
    assert isinstance(config, Config)
    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    cli()
