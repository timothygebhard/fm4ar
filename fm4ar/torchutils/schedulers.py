"""
Utilities for PyTorch learning rate schedulers.
"""

from importlib import import_module
from typing import Any, Callable, Literal, Type

import torch
from pydantic import BaseModel, Field, field_validator
from torch.optim import lr_scheduler as lrs

from fm4ar.torchutils.optimizers import get_lr


# Define joint type for learning rate schedulers
Scheduler = lrs.LRScheduler | lrs.ReduceLROnPlateau


class SchedulerConfig(BaseModel):
    """
    Configuration for a learning rate scheduler.
    """

    type: str = Field(
        ...,
        description=(
            "Name of the LR scheduler class. Must be a valid LR scheduler "
            "from `torch.optim.lr_scheduler`."
        ),
    )
    kwargs: dict = Field(
        {},
        description=(
            "Keyword arguments for the optimizer (e.g., patience)."
        ),
    )

    @field_validator('type')
    def check_optimizer_type(cls: Any, v: str) -> str:
        try:
            getattr(import_module("torch.optim.lr_scheduler"), v)
        except AttributeError as e:
            raise ValueError(f"Invalid scheduler type: `{v}`") from e
        return v


def get_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    scheduler_config: SchedulerConfig,
) -> Scheduler:
    """
    Builds and returns a learning rate scheduler for `optimizer`.

    Args:
        optimizer: Optimizer for which the scheduler is used.
        scheduler_config: Configuration for the scheduler. Must contain
            the key `type` (e.g., "ReduceLROnPlateau") and `kwargs`.

    Returns:
        Learning rate scheduler for the optimizer.
    """

    # Get scheduler class based on the type
    # This should not require further error handling as the optimizer type
    # is already validated in the `OptimizerConfig` class
    SchedulerClass: Type[lrs.LRScheduler | lrs.ReduceLROnPlateau] = getattr(
        import_module("torch.optim.lr_scheduler"),
        scheduler_config.type,
    )

    # Instantiate scheduler from scheduler type and keyword arguments
    return SchedulerClass(optimizer, **scheduler_config.kwargs)


def perform_scheduler_step(
    scheduler: Scheduler,
    loss: float | None = None,
    end_of: Literal["epoch", "batch"] = "epoch",
    on_lower: Callable[[], Any] | None = None,
) -> None:
    """
    Wrapper for `scheduler.step()`. If scheduler is `ReduceLROnPlateau`,
    then `scheduler.step(loss)` is called, if not, `scheduler.step()`.

    Args:
        scheduler: Scheduler for learning rate.
        loss: Validation loss (only required for `ReduceLROnPlateau`).
        end_of: Whether the scheduler is called at the end of an epoch
            or at the end of a batch.
        on_lower: Callback function that is called when the learning
            rate is lowered (only supported for `ReduceLROnPlateau`).
            If `None`, no callback is called.
    """

    if end_of == "batch":

        # CyclicLR and OneCycleLR need to be called at the end of each batch
        end_of_batch_schedulers = (lrs.CyclicLR, lrs.OneCycleLR)
        if isinstance(scheduler, end_of_batch_schedulers):
            scheduler.step()

    elif end_of == "epoch":

        # StepLR, CosineAnnealingLR and CosineAnnealingWarmRestarts need
        # to be called at the end of each epoch
        end_of_epoch_schedulers = (
            lrs.CosineAnnealingLR,
            lrs.CosineAnnealingWarmRestarts,
            lrs.StepLR,
        )
        if isinstance(scheduler, end_of_epoch_schedulers):
            scheduler.step()

        # ReduceLROnPlateau requires special treatment as it needs to be
        # called with the validation loss. It is also the only scheduler
        # that supports a callback function that is called every time the
        # learning rate is lowered.
        if isinstance(scheduler, lrs.ReduceLROnPlateau):

            if loss is None:
                raise ValueError("Must provide loss for ReduceLROnPlateau!")

            old_lr = get_lr(scheduler.optimizer)[0]
            scheduler.step(loss)
            new_lr = get_lr(scheduler.optimizer)[0]
            if new_lr < old_lr and on_lower is not None:
                on_lower()

    else:
        raise ValueError("Invalid value for `end_of`!")
