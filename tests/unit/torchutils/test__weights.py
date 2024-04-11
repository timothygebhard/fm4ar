"""
Unit tests for `fm4ar.torchutils.weights`.
"""

from pathlib import Path

import torch

from fm4ar.torchutils.weights import (
    get_weights_from_pt_file,
    load_and_or_freeze_model_weights,
)


def test__get_weights_from_checkpoint(tmp_path: Path) -> None:
    """
    Test `fm4ar.torchutils.weights.get_weights_from_checkpoint()`.
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
    Test `fm4ar.torchutils.weights.load_and_or_freeze_model_weights()`.
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
