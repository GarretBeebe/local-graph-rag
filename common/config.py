"""Shared YAML config loading helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path, *, allow_empty: bool = False) -> dict[str, Any]:
    """Load a YAML config file as a dict.

    Raises the underlying file/YAML exceptions. Raises ValueError for empty configs
    unless allow_empty is true.
    """
    try:
        with path.open() as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Config file contains invalid YAML — {path}: {e}") from e
    if config is None:
        if allow_empty:
            return {}
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config
