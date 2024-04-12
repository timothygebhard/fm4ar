"""
Utilities for configuring early stopping.
"""

from typing import Sequence

import numpy as np
from pydantic import BaseModel, Field


class EarlyStoppingConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Whether to use early stopping.",
    )
    global_patience: int | None = Field(
        default=None,
        description="Number of (global) epochs to wait for improvement.",
    )
    stage_patience: int | None = Field(
        default=None,
        description="Number of (stage) epochs to wait for improvement.",
    )
    global_offset: int | None = Field(
        default=None,
        description=(
            "Number of epochs to wait before enabling global patience. "
            "Useful, e.g., when using a LR scheduler with a warm-up phase."
        ),
    )
    stage_offset: int | None = Field(
        default=None,
        description=(
            "Number of epochs to wait before enabling stage patience. "
            "Useful, e.g., when using a LR scheduler with a warm-up phase."
        ),
    )


def early_stopping_criterion_reached(
    loss_history: Sequence[float],
    stage_epoch: int,
    early_stopping_config: EarlyStoppingConfig,
) -> bool:
    """
    Check whether the early stopping criterion is reached.

    Args:
        loss_history: History of test losses.
        stage_epoch: Current epoch within the stage.
        early_stopping_config: Configuration for early stopping.

    Returns:
        Whether the early stopping criterion is reached.
    """

    # Check stage-based early stopping
    if early_stopping_config.stage_patience is not None:

        # We only need to look at the loss history from the current stage,
        # and we can drop the first `stage_offset` epochs
        loss_values = loss_history[-min(stage_epoch, len(loss_history)):]
        loss_values = loss_values[early_stopping_config.stage_offset:]

        # Check if the patience has been exceeded
        min_idx = int(np.argmin(loss_values))
        last_idx = len(loss_values)
        if last_idx - min_idx > early_stopping_config.stage_patience:
            return True

    # Check global early stopping
    if early_stopping_config.global_patience is not None:

        # In this case, we look at the full history, except the offset
        loss_values = loss_history[early_stopping_config.global_offset:]

        # Check if the patience has been exceeded
        min_idx = int(np.argmin(loss_values))
        last_idx = len(loss_values)
        if last_idx - min_idx > early_stopping_config.global_patience:
            return True

    return False
