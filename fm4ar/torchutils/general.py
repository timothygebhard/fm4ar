"""
General utility functions for PyTorch.
"""

from importlib import import_module
from math import prod
from typing import Any, Type

import torch
import torch.nn as nn

from fm4ar.nn.modules import Sine


def check_for_nans(x: torch.Tensor, label: str = "tensor") -> None:
    """
    Check if the given tensor (usually the loss) contains any entries
    that are NaN or infinite. If so, raise a `ValueError`.
    """

    if torch.isnan(x).any():
        raise ValueError(f"NaN values detected in {label}, aborting!")
    if torch.isinf(x).any():
        raise ValueError(f"Inf values detected in {label}, aborting!")


def get_activation_from_name(name: str) -> torch.nn.Module:
    """
    Build and return an activation function with the given name.
    """

    ActivationFunction: Type[torch.nn.Module]

    # Custom activation functions need special treatment
    if name == "Sine":
        ActivationFunction = Sine

    # All other activation functions are built from `torch.nn`
    else:
        try:
            ActivationFunction = getattr(import_module("torch.nn"), name)
        except AttributeError as e:
            raise ValueError(f"Invalid activation function: `{name}`") from e

    # Instantiate the activation function and return it
    return ActivationFunction()


def get_cuda_info() -> dict[str, Any]:
    """
    Get information about the CUDA devices available in the system.
    """

    # No CUDA devices available
    if not torch.cuda.is_available():
        return {}

    # CUDA devices are available
    return {
        "cuDNN version": torch.backends.cudnn.version(),  # type: ignore
        "CUDA version": torch.version.cuda,
        "device count": torch.cuda.device_count(),
        "device name": torch.cuda.get_device_name(0),
        "memory (GB)": round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
        ),
    }


def get_number_of_parameters(
    model: nn.Module,
    requires_grad_flags: tuple[bool, ...] = (True, False),
) -> int:
    """
    Count the number of parameters of the given `model`.

    Args:
        model: Model for which to count the number of parameters.
        requires_grad_flags: Tuple of bools that specify which values
            of `requires_grad` should be counted.

    Returns:
        Number of parameters of the model.
    """

    num_params = 0
    for p in list(model.parameters()):
        if p.requires_grad in requires_grad_flags:
            num_params += prod(p.size())
    return num_params


def resolve_device(device: str) -> torch.device:
    """
    Resolve the device string to a `torch.device` object.
    """

    # "auto" means "cuda" if available, otherwise "cpu"
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Otherwise, just return the device as specified
    return torch.device(device)


def set_random_seed(seed: int) -> None:
    """
    Set the seed for all PyTorch-related random number generators.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
