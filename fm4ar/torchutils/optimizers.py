"""
Utilities for PyTorch optimizers.
"""

from importlib import import_module
from typing import Any, Iterable, Type

import torch
from pydantic import BaseModel, Field, field_validator


class OptimizerConfig(BaseModel):
    """
    Configuration for an optimizer.
    """

    type: str = Field(
        ...,
        description=(
            "Type of the optimizer. Must be a valid optimizer "
            "from `torch.optim`."
        ),
    )
    kwargs: dict = Field(
        {},
        description=(
            "Keyword arguments for the optimizer (e.g., learning rate)."
        ),
    )

    @field_validator('type')
    def check_optimizer_type(cls: Any, v: str) -> str:
        try:
            getattr(import_module("torch.optim"), v)
        except AttributeError as e:
            raise ValueError(f"Invalid optimizer type: `{v}`") from e
        return v


def get_optimizer_from_config(
    model_parameters: Iterable,
    optimizer_config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """
    Builds and returns an optimizer for `model_parameters`, based on
    the given `optimizer_config`.

    This should support all optimizers from `torch.optim`.
    Note: Some optimizers, such as L-BFGS, can be specified like this
    but will not work for training as they require the `closure` to be
    defined (see the PyTorch documentation for details).

    Args:
        model_parameters: Model parameters to optimize.
        optimizer_config: Configuration for the optimizer.

    Returns:
        Optimizer for the model parameters.
    """

    # Get optimizer class based on the type
    # This should not require further error handling as the optimizer type
    # is already validated in the `OptimizerConfig` class
    OptimizerClass: Type[torch.optim.Optimizer] = getattr(
        import_module("torch.optim"),
        optimizer_config.type,
    )

    # Instantiate optimizer from optimizer type and keyword arguments
    return OptimizerClass(model_parameters, **optimizer_config.kwargs)


def get_lr(optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Returns a list with the learning rates of the optimizer.
    """

    return [
        float(param_group["lr"])
        for param_group in optimizer.state_dict()["param_groups"]
    ]
