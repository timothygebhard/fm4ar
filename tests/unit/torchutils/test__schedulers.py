"""
Unit tests for `fm4ar.torchutils.schedulers`.
"""

import pytest
import torch
import torch.optim.lr_scheduler as lrs

from fm4ar.torchutils.optimizers import get_lr
from fm4ar.torchutils.schedulers import (
    Scheduler,
    SchedulerConfig,
    get_scheduler_from_config,
    perform_scheduler_step,
)


def test__scheduler_config() -> None:
    """
    Test `fm4ar.torchutils.schedulers.SchedulerConfig`.
    """

    # Case 1: Valid scheduler type
    scheduler_config = SchedulerConfig(
        type="StepLR",
        kwargs={"step_size": 1, "gamma": 0.1},
    )
    assert scheduler_config.type == "StepLR"
    assert scheduler_config.kwargs == {"step_size": 1, "gamma": 0.1}

    # Case 2: Another valid scheduler type
    scheduler_config = SchedulerConfig(
        type="ReduceLROnPlateau",
        kwargs={"mode": "min", "factor": 0.1},
    )
    assert scheduler_config.type == "ReduceLROnPlateau"
    assert scheduler_config.kwargs == {"mode": "min", "factor": 0.1}

    # Case 3: Invalid scheduler type
    with pytest.raises(ValueError) as value_error:
        SchedulerConfig(
            type="invalid",
            kwargs={"mode": "min", "factor": 0.1},
        )
    assert "Invalid scheduler type" in str(value_error)


def test__get_scheduler_from_config() -> None:
    """
    Test `fm4ar.torchutils.schedulers.get_scheduler_from_config()`.
    """

    # Define dummy model and optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Case 1: StepLR scheduler
    scheduler_config = SchedulerConfig(
        type="StepLR",
        kwargs={"step_size": 1, "gamma": 0.1},
    )
    scheduler = get_scheduler_from_config(optimizer, scheduler_config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 1

    # Case 2: ReduceLROnPlateau scheduler
    scheduler_config = SchedulerConfig(
        type="ReduceLROnPlateau",
        kwargs={"mode": "min", "factor": 0.1},
    )
    scheduler = get_scheduler_from_config(optimizer, scheduler_config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "min"
    assert scheduler.factor == 0.1


@pytest.mark.filterwarnings("ignore:Detected call of")
def test__perform_scheduler_step() -> None:
    """
    Test `perform_scheduler_step()`.
    """

    # Create dummy optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler: Scheduler

    # Case 1: Scheduler that is called at the end of each batch
    scheduler = lrs.CyclicLR(optimizer, base_lr=0.1, max_lr=1)
    initial_lr = scheduler.get_last_lr()
    perform_scheduler_step(scheduler, end_of="batch")
    updated_lr = scheduler.get_last_lr()
    assert updated_lr != initial_lr
    perform_scheduler_step(scheduler, end_of="epoch")
    assert scheduler.get_last_lr() == updated_lr

    # Case 2: Scheduler that is called at the end of each epoch
    scheduler = lrs.StepLR(optimizer, step_size=1, gamma=0.5)
    initial_lr = scheduler.get_last_lr()
    perform_scheduler_step(scheduler, end_of="epoch")
    updated_lr = scheduler.get_last_lr()
    assert updated_lr != initial_lr
    perform_scheduler_step(scheduler, end_of="batch")
    assert scheduler.get_last_lr() == updated_lr

    # Prepare a flag to check if the `on_lower()` method is called
    on_lower_flag = False

    def on_lower() -> None:
        nonlocal on_lower_flag
        on_lower_flag = True

    # Case 3: ReduceLROnPlateau scheduler
    scheduler = lrs.ReduceLROnPlateau(optimizer, patience=0)
    initial_lr = get_lr(scheduler.optimizer)
    with pytest.raises(ValueError) as value_error:
        perform_scheduler_step(scheduler, end_of="epoch")
    assert "Must provide loss for ReduceLROnPlateau!" in str(value_error)
    perform_scheduler_step(
        scheduler, end_of="epoch", loss=0.1, on_lower=on_lower
    )
    updated_lr = get_lr(scheduler.optimizer)
    assert updated_lr == initial_lr  # first step should not change LR
    assert not on_lower_flag
    perform_scheduler_step(
        scheduler, end_of="epoch", loss=0.2, on_lower=on_lower
    )
    updated_lr = get_lr(scheduler.optimizer)
    assert updated_lr != initial_lr  # second step should change LR
    assert on_lower_flag

    # Case 4: Invalid end_of argument
    with pytest.raises(ValueError) as value_error:
        # noinspection PyTypeChecker
        perform_scheduler_step(scheduler, end_of="invalid")
    assert "Invalid value for" in str(value_error)

    # Case 5: Invalid scheduler type
    scheduler = object()
    with pytest.raises(ValueError) as value_error:
        perform_scheduler_step(scheduler, end_of="epoch")
    assert "Unknown scheduler type" in str(value_error)
