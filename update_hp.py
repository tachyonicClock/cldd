import click
import optuna
import os
from pathlib import Path
import re

from src.util import dict_dot_notation_set
import yaml


@click.command()
@click.argument("study_name", type=str)
def main(study_name: str):
    storage = os.environ.get("OPTUNA_STORAGE")
    assert storage is not None, "OPTUNA_STORAGE environment variable must be set."

    # bocl/hp/RotatedTinyMNIST_0.0/EWC_oracle_MLP
    # bocl/{}/{}/{}_{}_{}
    # Extract label, scenario, learner, drift detector, and model from the study name using regex.
    match = re.match(r"bocl/([^/]+)/([^/]+)/([^/]+)", study_name)
    if match:
        label, boundary, method = match.groups()
        learner, drift_detector, model = method.split("_")
    else:
        raise ValueError(f"Study name `{study_name}` does not match expected format.")

    filename = (Path("config") / boundary / method).with_suffix(".yml")
    print(f"Update `{filename}` with best trial from study `{study_name}`.")
    study = optuna.load_study(study_name=study_name, storage=storage)

    obj = {}
    obj["bases"] = [
        "scenario.yml",
        f"boundary/{boundary}.yml",
        f"learner/{learner}.yml",
        f"drift_detector/{drift_detector}.yml",
    ]

    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        dict_dot_notation_set(key, obj, value)

    base_study_params = study.user_attrs.get("base_study.best_trial.params", {})
    for key, value in base_study_params.items():
        dict_dot_notation_set(key, obj, value)

    with open(filename, "w") as f:
        yaml.dump(obj, f)


if __name__ == "__main__":
    main()
