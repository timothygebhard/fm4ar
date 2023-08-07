"""
Method for loading experiment configuration files.
"""

from pathlib import Path

import yaml


def load_config(
    experiment_dir: Path,
    file_name: str = "config.yaml",
) -> dict:
    """
    Load the configuration file for an experiment.

    Args:
        experiment_dir: Path to the experiment directory.
        file_name: Name of the config file (default: "config.yaml").

    Returns:
        Dictionary containing the configuration settings.
    """

    file_path = experiment_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"No config.yaml in {experiment_dir}!")

    with open(file_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return dict(config)


def save_config(
    config: dict,
    experiment_dir: Path,
    file_name: str = "local_config.yaml",
) -> None:
    """
    Save a configuration dictionary to a YAML file.
    Usually, this is used to save the *local* configuration.

    Args:
        config: Dictionary containing the configuration settings.
        experiment_dir: Path to the experiment directory.
        file_name: Config file name (default: "local_config.yaml").
    """

    file_path = experiment_dir / file_name
    with open(file_path, "w") as yaml_file:
        yaml.dump(
            config,
            yaml_file,
            default_flow_style=False,
            sort_keys=False,
        )
