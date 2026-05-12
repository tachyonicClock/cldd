from dataclasses import dataclass
from typing import Literal, Optional
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import cattrs
from cattrs.strategies import configure_tagged_union
from src.scenario import Scenario




@dataclass
class Config:
    name: str
    scenario: Scenario
    bases: Optional[list[str]] = None

# Setup cattrs converter for auto-disambiguation of union types.
_converter = cattrs.Converter()
_converter.forbid_extra_keys = True
configure_tagged_union(StrategyA | StrategyB, _converter)


def get_config(config_path: Path, bases_dir: Path, dotlist: list[str]) -> Config:
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a DictConfig, got {type(config)}")

    # Merge with precedence bases < config < dotlist
    bases = [OmegaConf.load(bases_dir / base) for base in config.get("bases", [])]
    bases.append(config)
    bases.append(OmegaConf.from_dotlist(dotlist))

    return _converter.structure(OmegaConf.merge(*bases), Config)
