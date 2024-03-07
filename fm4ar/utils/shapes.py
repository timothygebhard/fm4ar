"""
Utilities for validating shapes of tensors and arrays.
"""

import numpy as np
import torch


def validate_dims(
    x: torch.Tensor | np.ndarray,
    ndim: int,
) -> None:
    """
    Validate that `x` has the correct number of dimensions.

    Raises:
        ValueError: If `x.ndim != ndim`.

    Args:
        x: A torch tensor or numpy array.
        ndim: The expected number of dimensions.
    """

    if x.ndim != ndim:
        name = f"{x=}".split("=")[0].strip()
        raise ValueError(
            f"Expected `{name}` to have {ndim} dimensions but found {x.ndim}!"
        )


def validate_shape(
    x: torch.Tensor | np.ndarray,
    shape: tuple[int | None, ...],
) -> None:
    """
    Validate that `x` has the correct shape.

    Args:
        x: A torch tensor or numpy array.
        shape: The expected shape. `None` means that the dimension can
            have any size.
    """

    # Check if the number of dimensions is correct
    validate_dims(x=x, ndim=len(shape))

    # Check if the size of each dimension is correct
    for expected, actual in zip(shape, x.shape, strict=True):
        if expected is not None and expected != actual:
            name = f"{x=}".split("=")[0].strip()
            raise ValueError(
                f"Expected `{name}` to have shape {shape} but found {x.shape}!"
            )
