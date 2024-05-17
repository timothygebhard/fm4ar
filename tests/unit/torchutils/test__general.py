"""
Unit tests for `fm4ar.torchutils.general`.
"""

from types import SimpleNamespace
from typing import Type

import pytest
import torch

from fm4ar.nn.modules import Sine
from fm4ar.torchutils.general import (
    check_for_nans,
    get_activation_from_name,
    get_cuda_info,
    get_number_of_parameters,
    resolve_device,
)


def test__check_for_nans() -> None:
    """
    Test `fm4ar.torchutils.general.check_for_nans()`.
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
    Test `fm4ar.torchutils.general.get_activation_from_string()`.
    """

    if activation_name == "invalid":
        with pytest.raises(ValueError) as value_error:
            get_activation_from_name(activation_name)
        assert "Invalid activation function" in str(value_error)

    else:
        activation = get_activation_from_name(activation_name)
        assert isinstance(activation, expected_activation)


def test__get_cuda_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test `fm4ar.torchutils.general.get_cuda_info()`.
    """

    # Case 1: No CUDA devices available
    with monkeypatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: False)
        cuda_info = get_cuda_info()
        assert cuda_info == {}

    # Case 2: Pretend we have a CUDA device
    with monkeypatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: True)
        mp.setattr("torch.backends.cudnn.version", lambda: 123)
        mp.setattr("torch.version.cuda", "11.1")
        mp.setattr("torch.cuda.device_count", lambda: 1)
        mp.setattr("torch.cuda.get_device_name", lambda _: "GeForce GTX 1080")
        mp.setattr(
            "torch.cuda.get_device_properties",
            lambda _: SimpleNamespace(total_memory=8 * 1024 ** 3),
        )
        cuda_info = get_cuda_info()
        assert cuda_info == {
            "cuDNN version": 123,
            "CUDA version": "11.1",
            "device count": 1,
            "device name": "GeForce GTX 1080",
            "memory (GB)": 8.0,
        }


def test__get_number_of_parameters() -> None:
    """
    Test `fm4ar.torchutils.general.get_number_of_parameters()`.
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


def test__resolve_device() -> None:
    """
    Test `fm4ar.torchutils.general.resolve_device()`.
    """

    # Case 1: "cpu"
    device = resolve_device("cpu")
    assert device == torch.device("cpu")

    # Case 2: "cuda"
    device = resolve_device("cuda")
    assert device == torch.device("cuda")

    # Case 3: "auto" with cuda available
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: True)
        device = resolve_device("auto")
        assert device == torch.device("cuda")

    # Case 4: "auto" without cuda available
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("torch.cuda.is_available", lambda: False)
        device = resolve_device("auto")
        assert device == torch.device("cpu")

    # Case 5: Invalid device
    with pytest.raises(RuntimeError) as runtime_error:
        resolve_device("invalid")
    assert "Expected one of" in str(runtime_error)
