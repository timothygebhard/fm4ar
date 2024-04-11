"""
Utilities for building PyTorch `DataLoader` objects.
"""

import platform

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from fm4ar.utils.multiproc import get_number_of_available_cores


def build_dataloaders(
    dataset: Dataset,
    n_train_samples: int,
    n_valid_samples: int,
    batch_size: int,
    n_workers: int,
    drop_last: bool = True,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation `DataLoaders` for the given `dataset`.

    Args:
        dataset: Full dataset to be split into train and test sets.
        n_train_samples: Number of samples to use for training.
        n_valid_samples: Number of samples to use for validation.
        batch_size: Batch size for the train and test loaders.
        n_workers: Number of workers for the train and test loaders.
        drop_last: Whether to drop the last batch if it is smaller than
            `batch_size`. This is only used for the train loader.
        random_seed: Random seed for reproducibility.

    Returns:
        A 2-tuple: `(train_loader, valid_loader)`.
    """

    # Split the dataset into train and validation sets
    train_dataset, valid_dataset = random_split(
        dataset=dataset,
        lengths=[n_train_samples, n_valid_samples],
        generator=torch.Generator().manual_seed(random_seed + 0),
    )

    # Build the train loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=n_workers,
        generator=torch.Generator().manual_seed(random_seed + 1),
    )

    # Build the validation loader
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=n_workers,
        generator=torch.Generator().manual_seed(random_seed + 2),
    )

    return train_loader, valid_loader


def get_number_of_workers(n_workers: int | str) -> int:
    """
    Determine the number of workers for a `DataLoader`.

    Args:
        n_workers: If an integer is given, this is returned. If "auto"
            is given, we determine the number of cores based on the
            host system. Any other value raises a `ValueError`.

    Returns:
        Number of workers for the `DataLoader`.
    """

    # If an explicit number of workers is given, return it
    if isinstance(n_workers, int):
        return n_workers

    # If we are running locally on a Mac, the number of workers needs to be 0
    if platform.system() == "Darwin":
        return 0

    # Otherwise, use all but one available cores (but at least one)
    if n_workers == "auto":
        n_available_cores = get_number_of_available_cores()
        return max(n_available_cores - 1, 1)

    raise ValueError("Invalid value for `n_workers`!")
