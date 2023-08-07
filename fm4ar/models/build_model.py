"""
Method to build (i.e., instantiate) a model from a checkpoint file or
from a configuration dictionary.
"""

from pathlib import Path
from typing import Any

import torch

from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.models.discrete.normalizing_flow import NormalizingFlow


def build_model(
    file_path: Path | None = None,
    config: dict | None = None,
    **kwargs: Any,
) -> FlowMatching | NormalizingFlow:
    """
    Build a model from a checkpoint file or from a `config` dictionary.

    Args:
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
        case "flow_matching":
            return FlowMatching(
                file_path=file_path,
                config=config,
                **kwargs,
            )
        case "neural_posterior_estimation":
            return NormalizingFlow(
                file_path=file_path,
                config=config,
                **kwargs,
            )
        case _:
            raise ValueError(f"{model_type} is not a valid model type!")
