"""
Loads config.yaml: SLURM/AlphaFold3 defaults and absolute paths to external data.

config.yaml holds tracked placeholder defaults; config.local.yaml (gitignored, same
directory) holds personal values and overrides config.yaml key by key.

Ben Iovino  09/29/26    Biohub
"""

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"
LOCAL_CONFIG_FILE = CONFIG_FILE.with_name("config.local.yaml")


def _merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base (nested dicts merged, other values replaced).

    Args:
        base (dict): Default configuration.
        override (dict): Values that take precedence over base.

    Returns:
        dict: The merged configuration.
    """

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value

    return merged


with open(CONFIG_FILE, "r") as _f:
    CONFIG = yaml.safe_load(_f)
if LOCAL_CONFIG_FILE.exists():
    with open(LOCAL_CONFIG_FILE, "r") as _f:
        CONFIG = _merge(CONFIG, yaml.safe_load(_f) or {})

PATHS = {k: Path(v) for k, v in CONFIG["paths"].items()}
