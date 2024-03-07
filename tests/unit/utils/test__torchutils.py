"""
Unit tests for `fm4ar.utils.torchutils`.
"""

from pathlib import Path
from typing import Type

import numpy as np
import pytest
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

from fm4ar.nn.modules import Sine
from fm4ar.utils.torchutils import (
    build_train_and_test_loaders,
    check_for_nans,
    get_activation_from_name,
    get_lr,
    get_mlp,
    get_number_of_parameters,
    get_optimizer_from_config,
    get_scheduler_from_config,
    get_weights_from_pt_file,
    load_and_or_freeze_model_weights,
    perform_scheduler_step,
    split_dataset_into_train_and_test,
)


@pytest.fixture
def dummy_dataset() -> TensorDataset:
    """
    Create a dummy dataset for testing.
    """

    # Create a dummy dataset
    x = torch.linspace(0, 1, 10)
    y = torch.arange(10) % 2

    return TensorDataset(x, y)


# We run this test for different numbers of workers to check if the loaders
# behave reproducibly both when running in the main thread (0 workers) and
# when running in parallel (1 worker).
@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.slow  # because of num_workers=1
def test__build_train_and_test_loaders(
    num_workers: int,
    dummy_dataset: TensorDataset,
) -> None:
    """
    Test `build_train_and_test_loaders()`.
    """

    # Check that we can create dataloaders with the expected lengths
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dummy_dataset,
        train_fraction=0.5,
        batch_size=2,
        num_workers=num_workers,
        drop_last=True,
        random_seed=42,
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert len(train_loader) == 2  # last batch is dropped
    assert len(test_loader) == 3  # last batch is not dropped
    assert len(train_loader.dataset) == 5  # type: ignore
    assert len(test_loader.dataset) == 5  # type: ignore

    # Check that the dataloaders are reproducible
    assert np.isclose(torch.sum(next(iter(train_loader))[0]).item(), 1.0)
    assert np.isclose(torch.sum(next(iter(test_loader))[0]).item(), 5 / 9)

    # Check that we do not always get the same batch
    old_batch = next(iter(train_loader))
    for _ in range(10):
        new_batch = next(iter(train_loader))
        assert not torch.allclose(old_batch[0], new_batch[0])
        old_batch = new_batch


def test__check_for_nans() -> None:
    """
    Test `check_for_nans()`.
    """

    # Case 1: No NaN in the tensor
    tensor = torch.tensor([1.0, 2.0, 3.0])
    check_for_nans(tensor)

    # Case 2: NaN in the tensor
    tensor = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError) as value_error:
        check_for_nans(tensor, "my_tensor")
    assert "NaN values detected in my_tensor" in str(value_error)

    # Case 3: inf in the tensor
    tensor = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError) as value_error:
        check_for_nans(tensor, "my_tensor")
    assert "Inf values detected in my_tensor" in str(value_error)

    # Case 4: -inf in the tensor
    tensor = torch.tensor([1.0, float("-inf"), 3.0])
    with pytest.raises(ValueError) as value_error:
        check_for_nans(tensor, "my_tensor")
    assert "Inf values detected in my_tensor" in str(value_error)


@pytest.mark.parametrize(
    "activation_name, expected_activation",
    [
        ("ELU", torch.nn.ELU),
        ("GELU", torch.nn.GELU),
        ("LeakyReLU", torch.nn.LeakyReLU),
        ("ReLU", torch.nn.ReLU),
        ("Sigmoid", torch.nn.Sigmoid),
        ("Sine", Sine),
        ("SiLU", torch.nn.SiLU),  # same as Swish
        ("Tanh", torch.nn.Tanh),
        ("invalid", None),
    ],
)
def test__get_activation_from_string(
    activation_name: str,
    expected_activation: Type[torch.nn.Module],
) -> None:
    """
    Test `get_activation_from_string()`.
    """

    if activation_name == "invalid":
        with pytest.raises(ValueError) as value_error:
            get_activation_from_name(activation_name)
        assert "Invalid activation function" in str(value_error)

    else:
        activation = get_activation_from_name(activation_name)
        assert isinstance(activation, expected_activation)


def test__get_mlp() -> None:
    """
    Test `get_mlp()`.
    """

    # Case 1: Single hidden layer, no dropout or batch norm
    mlp = get_mlp(
        input_dim=10,
        hidden_dims=[5],
        output_dim=1,
        activation="Tanh",
        dropout=0.0,
    )
    assert isinstance(mlp, torch.nn.Sequential)
    assert len(mlp) == 3
    assert isinstance(mlp[0], torch.nn.Linear)
    assert isinstance(mlp[1], torch.nn.Tanh)
    assert isinstance(mlp[2], torch.nn.Linear)
    assert mlp(torch.randn(7, 10)).shape == (7, 1)

    # Case 2: Multiple hidden layers with dropout and batch norm
    mlp = get_mlp(
        input_dim=10,
        hidden_dims=[5, 5],
        output_dim=1,
        activation="SiLU",
        dropout=0.5,
        batch_norm=True,
    )
    assert isinstance(mlp, torch.nn.Sequential)
    assert len(mlp) == 9
    assert isinstance(mlp[0], torch.nn.Linear)
    assert isinstance(mlp[1], torch.nn.SiLU)
    assert isinstance(mlp[2], torch.nn.BatchNorm1d)
    assert isinstance(mlp[3], torch.nn.Dropout)
    assert isinstance(mlp[4], torch.nn.Linear)
    assert isinstance(mlp[5], torch.nn.SiLU)
    assert isinstance(mlp[6], torch.nn.BatchNorm1d)
    assert isinstance(mlp[7], torch.nn.Dropout)
    assert isinstance(mlp[8], torch.nn.Linear)
    assert mlp(torch.randn(7, 10)).shape == (7, 1)


def test__get_number_of_parameters() -> None:
    """
    Test `get_number_of_parameters()`.
    """

    layer_1 = torch.nn.Linear(10, 5)
    layer_1.requires_grad_(False)
    layer_2 = torch.nn.Linear(5, 1)
    model = torch.nn.Sequential(
        layer_1,
        torch.nn.ReLU(),
        layer_2,
    )

    n_trainable = get_number_of_parameters(model, (True,))
    n_fixed = get_number_of_parameters(model, (False,))
    n_total = get_number_of_parameters(model, (True, False))
    assert n_trainable == 6
    assert n_fixed == 55
    assert n_total == 61


@pytest.mark.parametrize(
    "optimizer_config, expected_class",
    [
        ({"type": "invalid"}, None),
        ({"type": "Adam", "kwargs": {"betas": [0.9, 0.95]}}, torch.optim.Adam),
        ({"type": "AdamW", "kwargs": {"lr": 4e-3}}, torch.optim.AdamW),
        ({"type": "SGD", "kwargs": {"lr": 0.1}}, torch.optim.SGD),
    ],
)
def test__get_optimizer_from_config(
    optimizer_config: dict,
    expected_class: Type[torch.optim.Optimizer],
) -> None:
    """
    Test `get_optimizer_from_kwargs()`.
    """

    model = torch.nn.Linear(10, 5)

    # Case 1: Invalid optimizer type
    if optimizer_config["type"] == "invalid":
        with pytest.raises(ValueError) as value_error:
            get_optimizer_from_config(
                model_parameters=model.parameters(),
                optimizer_config=optimizer_config,
            )
        assert "Invalid optimizer type" in str(value_error)

    # Case 2: Valid optimizer type
    else:
        optimizer = get_optimizer_from_config(
            model_parameters=model.parameters(),
            optimizer_config=optimizer_config,
        )
        assert isinstance(optimizer, expected_class)


@pytest.mark.parametrize(
    "scheduler_config, expected_class",
    [
        ({"type": "invalid"}, None),
        (
            {
                "type": "StepLR",
                "kwargs": {"step_size": 10},
            },
            lrs.StepLR,
        ),
        (
            {
                "type": "CosineAnnealingLR",
                "kwargs": {"T_max": 10},
            },
            lrs.CosineAnnealingLR,
        ),
        (
            {
                "type": "CosineAnnealingWarmRestarts",
                "kwargs": {"T_0": 10},
            },
            lrs.CosineAnnealingWarmRestarts,
        ),
        (
            {
                "type": "CyclicLR",
                "kwargs": {"base_lr": 0.1, "max_lr": 1},
            },
            lrs.CyclicLR,
        ),
        (
            {
                "type": "OneCycleLR",
                "kwargs": {"max_lr": 1, "total_steps": 5},
            },
            lrs.OneCycleLR,
        ),
        (
            {
                "type": "ReduceLROnPlateau",
                "kwargs": {"patience": 10},
            },
            lrs.ReduceLROnPlateau,
        ),
    ],
)
def test__get_scheduler_from_kwargs(
    scheduler_config: dict,
    expected_class: Type[lrs.LRScheduler | lrs.ReduceLROnPlateau],
) -> None:

    # Define dummy model and optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Case 1: Invalid scheduler type
    if scheduler_config["type"] == "invalid":
        with pytest.raises(ValueError) as value_error:
            get_scheduler_from_config(
                optimizer=optimizer,
                scheduler_config=scheduler_config,
            )
        assert "Invalid scheduler type" in str(value_error)

    # Case 2: Valid scheduler type
    else:
        scheduler = get_scheduler_from_config(
            optimizer=optimizer,
            scheduler_config=scheduler_config,
        )
        assert isinstance(scheduler, expected_class)


def test__get_weights_from_checkpoint(tmp_path: Path) -> None:
    """
    Test `get_weights_from_checkpoint()`.
    """

    # Create dummy model and save its state dict
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_1 = torch.nn.Linear(10, 5)
            self.layer_2 = torch.nn.Linear(5, 1)

    model = DummyModel()
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "model.pt")

    # Load state dict from checkpoint
    state_dict = get_weights_from_pt_file(
        file_path=tmp_path / "model.pt",
        state_dict_key="model_state_dict",
        prefix="layer_1",
        drop_prefix=False,
    )
    assert state_dict.keys() == {"layer_1.weight", "layer_1.bias"}
    assert torch.equal(state_dict["layer_1.weight"], model.layer_1.weight)
    assert torch.equal(state_dict["layer_1.bias"], model.layer_1.bias)


def test__load_and_or_freeze_model_weights(tmp_path: Path) -> None:
    """
    Test `load_and_or_freeze_model_weights()`.
    """

    # Create dummy model and save its state dict
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(10, 10, bias=False)

    model = DummyModel()
    model.layer.weight.data = torch.eye(10)
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "model.pt")

    # Re-create the model and load state dict from checkpoint
    model = DummyModel()
    load_and_or_freeze_model_weights(
        model=model,
        freeze_weights=True,
        load_weights={
            "file_path": str(tmp_path / "model.pt"),
            "state_dict_key": "model_state_dict",
            "prefix": "layer",
            "drop_prefix": False,
        },
    )

    # Check that the weights have been loaded and frozen
    assert torch.equal(model.layer.weight, torch.eye(10))
    assert not model.layer.weight.requires_grad


@pytest.mark.filterwarnings("ignore:Detected call of")
def test__perform_scheduler_step() -> None:
    """
    Test `perform_scheduler_step()`.
    """

    # Create dummy optimizer
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler: lrs.LRScheduler | lrs.ReduceLROnPlateau

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
    assert "Invalid value for `end_of`!" in str(value_error)


def test__split_dataset_into_train_and_test(
    dummy_dataset: TensorDataset,
) -> None:
    """
    Test `split_dataset_into_train_and_test()`.
    """

    train, test = split_dataset_into_train_and_test(
        dataset=dummy_dataset,
        train_fraction=0.5,
        random_seed=42,
    )
    assert isinstance(train, Subset)
    assert isinstance(test, Subset)
    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)
    assert len(train) == 5
    assert len(test) == 5
    assert train.indices == [2, 6, 1, 8, 4]
    assert test.indices == [5, 0, 9, 3, 7]
