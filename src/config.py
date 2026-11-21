"""Configuration loader — single source of truth for all runtime settings."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def _setup_logging(cfg: dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration as a plain dict.

    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml
                     relative to the project root.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            "Ensure configs/config.yaml exists in the project root."
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    _setup_logging(cfg)
    logger.info("Configuration loaded from '%s'", path)
    return cfg


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.parent.resolve()
