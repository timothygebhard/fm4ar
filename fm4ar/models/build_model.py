"""
Method to build (i.e., instantiate) a model from a checkpoint file or
from a configuration dictionary.
"""

from pathlib import Path
from typing import Any

import torch

from fm4ar.models.fmpe import FMPEModel
from fm4ar.models.npe import NPEModel


def build_model(
    experiment_dir: Path | None = None,
    file_path: Path | None = None,
    config: dict | None = None,
    **kwargs: Any,
) -> FMPEModel | NPEModel:
    """
    Build a model from a checkpoint file or from a `config` dictionary.

    Args:
        experiment_dir: Path to the experiment directory.
        file_path: Path to a checkpoint file (`*.pt`).
        config: Dictionary with full experiment configuration.
        **kwargs: Extra keyword arguments to pass to the model class.

    Returns:
        Model instance.
    """

    # Get the model type, either from the checkpoint file or from the settings
    if file_path is not None:
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        model_type = checkpoint["config"]["model"]["model_type"].lower()
    elif config is not None:
        model_type = config["model"]["model_type"].lower()
    else:
        raise ValueError("Either `file_path` or `config` must be provided!")

    # Select the model class
    match model_type:
        case "flow_matching" | "fm" | "fmpe":
            return FMPEModel(
                experiment_dir=experiment_dir,
                file_path=file_path,
                config=config,
                **kwargs,
            )
        case "neural_posterior_estimation" | "npe":
            return NPEModel(
                experiment_dir=experiment_dir,
                file_path=file_path,
                config=config,
                **kwargs,
            )
        case _:
            raise ValueError(f"{model_type} is not a valid model type!")
