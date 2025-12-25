"""Helper utilities for Credit Intelligence."""

import os
import logging
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(config_name: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file in config/ directory

    Returns:
        Configuration dictionary
    """
    config_path = get_project_root() / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Expand environment variables
    config = _expand_env_vars(config)

    return config


def _expand_env_vars(config: Any) -> Any:
    """Recursively expand environment variables in config."""
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(v) for v in config]
    elif isinstance(config, str):
        if config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        format_str: Optional custom format string

    Returns:
        Root logger
    """
    format_str = format_str or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers,
    )

    return logging.getLogger()


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency."""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B {currency}"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M {currency}"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K {currency}"
    else:
        return f"${value:.2f} {currency}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage."""
    return f"{value * 100:.2f}%"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def safe_get(data: Dict, *keys, default=None) -> Any:
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data
