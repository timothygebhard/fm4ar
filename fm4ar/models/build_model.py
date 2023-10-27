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
    experiment_dir: Path | None = None,
    file_path: Path | None = None,
    config: dict | None = None,
    update_config: bool = False,
    **kwargs: Any,
) -> FlowMatching | NormalizingFlow:
    """
    Build a model from a checkpoint file or from a `config` dictionary.

    Args:
        experiment_dir: Path to the experiment directory.
        file_path: Path to a checkpoint file (`*.pt`).
        config: Dictionary with full experiment configuration.
        update_config: If `True`, replace the configuration in the model
            checkpoint with the configuration from the experiment with
            the `config` passed to this function. This can be useful for
            small bug fixes; however, if the updated configuration is
            not compatible with the model in the checkpoint, this will
            lead to an error.
        **kwargs: Extra keyword arguments to pass to the model class.

    Returns:
        Model instance.
    """

    # Get the model type, either from the checkpoint file or from the settings
    if file_path is not None:
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        model_type = checkpoint["config"]["model"]["model_type"].lower()

        # Special case: We want to resume from a checkpoint but update the
        # configuration in the checkpoint with the configuration from the
        # experiment with the `config` passed to this function. In this case,
        # we overwrite the checkpoint with the updated configuration, which
        # will then be read again by the `load_model()` method of the model.
        if update_config and config is not None:
            checkpoint["config"] = config
            torch.save(checkpoint, file_path)

    elif config is not None:
        model_type = config["model"]["model_type"].lower()
    else:
        raise ValueError("Either `file_path` or `config` must be provided!")

    # Select the model class
    match model_type:
        case "flow_matching" | "fm":
            return FlowMatching(
                experiment_dir=experiment_dir,
                file_path=file_path,
                config=config,
                **kwargs,
            )
        case "neural_posterior_estimation" | "npe":
            return NormalizingFlow(
                experiment_dir=experiment_dir,
                file_path=file_path,
                config=config,
                **kwargs,
            )
        case _:
            raise ValueError(f"{model_type} is not a valid model type!")
