"""
Define parsers for model configurations.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from yaml import safe_load


class InputFileConfig(BaseModel):
    """
    Parser class for the input files configuration.
    """

    file_path: Path = Field(
        ...,
        description=(
            "Path to the input file. This can be either a nested sampling "
            "posterior file, or an HDF result file from FMPE / NPE."
        ),
    )
    file_type: Literal["ns", "ml"] = Field(
        ...,
        description=(
            "Type of the input file. Either 'ns' for nested sampling "
            "or 'ml' for FMPE / NPE."
        ),
    )
    n_samples: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of samples to load from the file. If None, all samples "
            "are loaded. This is the default."
        ),
    )


class ModelConfig(BaseModel):
    """
    Parser class for the model configuration.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_type: Literal["unconditional_flow"] = Field(
        ...,
        description="Type of the model to use. Required for compatibility.",
    )
    flow_wrapper: dict[str, Any] = Field(
        ...,
        description="Keyword arguments for constructing unconditional flow.",
    )


class UnconditionalFlowConfig(BaseModel):
    """
    Parser class for the general unconditional flow configuration.
    """

    input_files: list[InputFileConfig] = Field(
        ...,
        description="List of input files to load samples from.",
    )
    model: ModelConfig = Field(
        ...,
        description="Configuration for the model.",
    )
    theta_scaler: dict[str, Any] = Field(
        ...,
        description="Configuration for the scaler for `theta`.",
    )
    training: dict[str, Any] = Field(
        ...,
        description="Configuration for the training.",
    )


def load_config(experiment_dir: Path) -> UnconditionalFlowConfig:
    """
    Load the unconditional flow configuration from a YAML file.

    Note: Despite the name, this is not only the configuration of the
    flow itself, but also the configuration for the input files (which
    constitute the training data) and the training process itself.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        UnconditionalFlowConfig: The parsed configuration.
    """

    with open(experiment_dir / "config.yaml", "r") as f:
        config = safe_load(f)

    return UnconditionalFlowConfig(**config)
