"""
Unit tests for `fm4ar.torchutils.dataloaders`.
"""

import pytest
import torch

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from fm4ar.torchutils.dataloaders import (
    build_dataloaders,
    get_number_of_workers,
)


@pytest.fixture
def dummy_dataset() -> TensorDataset:
    """
    Create a dummy dataset for testing.
    """

    # Create a dummy dataset
    x = torch.arange(80)
    y = torch.arange(80) % 2

    return TensorDataset(x, y)


# We run this test for different numbers of workers to check if the loaders
# behave reproducibly both when running in the main thread (0 workers) and
# when running in parallel (2 workers).
@pytest.mark.parametrize("n_workers", [0, 2])
@pytest.mark.slow  # because of num_workers=2
def test__build_dataloaders(
    n_workers: int,
    dummy_dataset: TensorDataset,
) -> None:
    """
    Test `fm4ar.torchutils.dataloaders.build_dataloaders()`.
    """

    # -------------------------------------------------------------------------
    # Case 1: random_seed = 42
    # -------------------------------------------------------------------------

    # Check that we can create dataloaders with the expected lengths
    train_loader, valid_loader = build_dataloaders(
        dataset=dummy_dataset,
        n_train_samples=64,
        n_valid_samples=16,
        batch_size=3,
        n_workers=n_workers,
        drop_last=True,
        random_seed=42,
    )
    assert isinstance(train_loader, DataLoader)
    assert isinstance(valid_loader, DataLoader)
    assert len(train_loader) == 21  # last batch is dropped
    assert len(valid_loader) == 6  # last batch is not dropped
    assert len(train_loader.dataset) == 64  # type: ignore
    assert len(valid_loader.dataset) == 16  # type: ignore

    # Check that the dataset split is reproducible
    assert sum(train_loader.dataset.indices) == 2431  # type: ignore
    assert sum(valid_loader.dataset.indices) == 729  # type: ignore

    # Make sure that the batch order of the train data is reproducible.
    expected = [  # fmt: off
        118, 87, 118, 141, 154, 119, 120, 122, 143, 47,
        68, 115, 86, 161, 109, 112, 85, 59, 111, 140, 190,
    ]  # fmt: on
    for i, batch in enumerate(train_loader):
        actual = torch.sum(torch.Tensor(batch[0])).item()
        assert np.isclose(actual, expected[i])

    # On the second "epoch", we should get new batches (not just new order)!
    expected = [  # fmt: off
        152, 128, 154, 122, 154, 78, 118, 111, 156, 198,
        70, 71, 98, 89, 79, 142, 116, 80, 113, 67, 64,
    ]  # fmt: on
    for i, batch in enumerate(train_loader):
        actual = torch.sum(torch.Tensor(batch[0])).item()
        assert np.isclose(actual, expected[i])

    # Check that the batch order for the validation loader is reproducible.
    # Here, the batches should be the same every epoch.
    expected = [195, 91, 73, 138, 167, 65]
    for _ in range(3):
        for i, batch in enumerate(valid_loader):
            actual = torch.sum(torch.Tensor(batch[0])).item()
            assert np.isclose(actual, expected[i])

    # -------------------------------------------------------------------------
    # Case 1: random_seed = 12345
    # -------------------------------------------------------------------------

    train_loader, valid_loader = build_dataloaders(
        dataset=dummy_dataset,
        n_train_samples=64,
        n_valid_samples=16,
        batch_size=3,
        n_workers=n_workers,
        drop_last=True,
        random_seed=12345,
    )

    # Different seed should mean different split
    assert sum(train_loader.dataset.indices) == 2586  # type: ignore
    assert sum(valid_loader.dataset.indices) == 574  # type: ignore

    # Check again the batch order of the train data
    expected = [  # fmt: off
        127, 56, 66, 123, 172, 133, 130, 89, 140, 130, 160,
        149, 165, 177, 147, 67, 84, 186, 69, 142, 54,
    ]  # fmt: on
    for i, batch in enumerate(train_loader):
        actual = torch.sum(torch.Tensor(batch[0])).item()
        assert np.isclose(actual, expected[i])

    # Again, we should get new batches for epoch number 2
    expected = [  # fmt: off
        213, 130, 144, 78, 127, 108, 125, 89, 210, 167, 93,
        93, 108, 209, 137, 81, 116, 88, 44, 109, 90,
    ]  # fmt: on
    for i, batch in enumerate(train_loader):
        actual = torch.sum(torch.Tensor(batch[0])).item()
        assert np.isclose(actual, expected[i])

    # Final check for the validation loader
    expected = [86, 179, 94, 100, 76, 39]
    for _ in range(3):
        for i, batch in enumerate(valid_loader):
            actual = torch.sum(torch.Tensor(batch[0])).item()
            assert np.isclose(actual, expected[i])


def test__get_number_of_workers() -> None:
    """
    Test `fm4ar.torchutils.dataloaders.get_number_of_workers()`.
    """

    # Case 1: Explicit number specified
    assert get_number_of_workers(123) == 123

    # Case 2: Assume we are on a Mac
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("platform.system", lambda: "Darwin")
        assert get_number_of_workers("auto") == 0

    # Case 3: Assume we are on a Linux machine
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("platform.system", lambda: "Linux")

        # We need to patch the function in torchutils, not in multiproc,
        # otherwise the patch has no effect because of the import order
        func = "fm4ar.torchutils.dataloaders.get_number_of_available_cores"

        mp.setattr(func, lambda: 12)
        assert get_number_of_workers("auto") == 11
        mp.setattr(func, lambda: 1)
        assert get_number_of_workers("auto") == 1

    # Case 4: Invalid input on a non
    with pytest.raises(ValueError) as value_error:
        get_number_of_workers("invalid")  # type: ignore
    assert str(value_error.value) == "Invalid value for `n_workers`!"
