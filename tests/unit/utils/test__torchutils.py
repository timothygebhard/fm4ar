"""
Unit tests for `fm4ar.utils.torchutils`.
"""

from pathlib import Path
from typing import Type

import pytest
import torch.nn

from fm4ar.nn.modules import Sine
from fm4ar.utils.torchutils import (
    get_activation_from_string,
    get_number_of_model_parameters,
    get_optimizer_from_kwargs,
    get_weights_from_pt_file,
    load_and_or_freeze_model_weights,
    validate_dims,
)


@pytest.mark.parametrize(
    "activation_name, expected_activation",
    [
        ("elu", torch.nn.ELU),
        ("gelu", torch.nn.GELU),
        ("leaky_relu", torch.nn.LeakyReLU),
        ("relu", torch.nn.ReLU),
        ("sigmoid", torch.nn.Sigmoid),
        ("sine", Sine),
        ("swish", torch.nn.SiLU),
        ("tanh", torch.nn.Tanh),
        ("invalid", None),
    ]
)
def test__get_activation_from_string(
    activation_name: str,
    expected_activation: Type[torch.nn.Module],
) -> None:
    """
    Test `fm4ar.utils.torchutils.get_activation_from_string()`.
    """

    if activation_name == "invalid":
        with pytest.raises(ValueError) as value_error:
            get_activation_from_string(activation_name)
        assert "Invalid activation function" in str(value_error)

    else:
        activation = get_activation_from_string(activation_name)
        assert isinstance(activation, expected_activation)


def test__get_number_of_model_parameters() -> None:
    """
    Test `fm4ar.utils.torchutils.get_number_of_model_parameters()`.
    """

    layer_1 = torch.nn.Linear(10, 5)
    layer_1.requires_grad_(False)
    layer_2 = torch.nn.Linear(5, 1)
    model = torch.nn.Sequential(
        layer_1,
        torch.nn.ReLU(),
        layer_2,
    )

    n_trainable = get_number_of_model_parameters(model, (True, ))
    n_fixed = get_number_of_model_parameters(model, (False, ))
    n_total = get_number_of_model_parameters(model, (True, False))
    assert n_trainable == 6
    assert n_fixed == 55
    assert n_total == 61


@pytest.mark.parametrize(
    "optimizer_type,expected_class",
    [
        ("adagrad", torch.optim.Adagrad),
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("lbfgs", torch.optim.LBFGS),
        ("rmsprop", torch.optim.RMSprop),
        ("sgd", torch.optim.SGD),
        ("invalid", None),
    ]
)
def test__get_optimizer_from_kwargs(
    optimizer_type: str,
    expected_class: Type[torch.optim.Optimizer],
) -> None:
    """
    Test `fm4ar.utils.torchutils.get_optimizer_from_kwargs()`.
    """

    if optimizer_type == "invalid":

        # Case 1: Missing optimizer type
        with pytest.raises(KeyError) as key_error:
            get_optimizer_from_kwargs(model_parameters=[])
        assert "Optimizer type needs to be specified!" in str(key_error)

        # Case 2: Invalid optimizer type
        with pytest.raises(ValueError) as value_error:
            get_optimizer_from_kwargs(
                model_parameters=[],
                type=optimizer_type,
            )
        assert "Invalid optimizer type" in str(value_error)

    else:

        # Case 3: Valid optimizer type
        model = torch.nn.Linear(10, 5)
        optimizer = get_optimizer_from_kwargs(
            model_parameters=model.parameters(),
            type=optimizer_type,
            lr=0.1,
        )
        assert isinstance(optimizer, expected_class)


def test__get_weights_from_checkpoint(tmp_path: Path) -> None:
    """
    Test `fm4ar.utils.torchutils.get_weights_from_checkpoint()`.
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
    state_dict = get_weights_from_pt_file(tmp_path / "model.pt", "layer_1")
    assert state_dict.keys() == {"layer_1.weight", "layer_1.bias"}
    assert torch.equal(state_dict["layer_1.weight"], model.layer_1.weight)
    assert torch.equal(state_dict["layer_1.bias"], model.layer_1.bias)


def test__load_and_or_freeze_model_weights(tmp_path: Path) -> None:
    """
    Test `fm4ar.utils.torchutils.load_and_or_freeze_model_weights()`.
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
            "prefix": "layer",
        }
    )

    # Check that the weights have been loaded and frozen
    assert torch.equal(model.layer.weight, torch.eye(10))
    assert not model.layer.weight.requires_grad


def test__validate_dims() -> None:
    """
    Test `fm4ar.utils.torchutils.validate_dims()`.
    """

    # Case 1: Correct dimensions
    x = torch.randn(10, 5)
    validate_dims(x=x, ndim=2)

    # Case 2: Incorrect dimensions
    x = torch.randn(10, 5)
    with pytest.raises(ValueError) as value_error:
        validate_dims(x=x, ndim=3)
    assert "Expected `x` to have 3 dimensions but found 2!" in str(value_error)
