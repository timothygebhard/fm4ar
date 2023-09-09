"""
Utility functions for PyTorch.
"""

from collections import OrderedDict
from math import prod
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.optim import lr_scheduler as lrs
from torch.utils.data import DataLoader


def get_activation_from_string(
    name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Map the name of an activation function name to the corresponding
    activation function from `torch.nn.functional`.

    Args:
        name: Name of the activation function.

    Returns:
        The activation function for the given name.
    """

    match name.lower():
        case "elu":
            return functional.elu
        case "relu":
            return functional.relu
        case "leaky_relu":
            return functional.leaky_relu
        case "gelu":
            return functional.gelu
        case _:
            raise ValueError("Invalid activation function!")


def forward_pass_with_unpacked_tuple(
    model: nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Performs forward pass of `model` with input `x`. If `x` is a tensor,
    it return `model(x)`, else it returns `model(*x)`.

    Args:
        model: Model for forward pass.
        x: Input for forward pass: Either a tensor or a tuple of
            tensors (the length of the tuple may be 0, though).

    Returns:
        Output of the forward pass, either `model(*x)` or `model(x)`.
    """

    return torch.Tensor(model(x) if isinstance(x, torch.Tensor) else model(*x))


def get_number_of_model_parameters(
    model: nn.Module,
    requires_grad_flags: tuple[bool, ...] = (True, False),
) -> int:
    """
    Count the number of parameters of the `model`.

    Args:
        model: Model for which to count the number of parameters.
        requires_grad_flags: Tuple of bools that specify which values
            of `requires_grad` should be counted.

    Returns:
        Number of parameters of the model.
    """

    num_params = 0
    for p in list(model.parameters()):
        if p.requires_grad in requires_grad_flags:
            num_params += prod(p.size())
    return num_params


def get_optimizer_from_kwargs(
    model_parameters: Iterable,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Builds and returns an optimizer for `model_parameters`, based on
    the `optimizer_kwargs`.

    Args:
        model_parameters: Model parameters to optimize.
        **optimizer_kwargs: Keyword arguments for the optimizer, such
            as the learning rate or momentum. Must also contain the key
            `type`, which specifies the optimizer type.

    Returns:
        Optimizer for the model parameters.
    """

    # Get optimizer type. We use `pop` to remove the key from the dictionary,
    # so that we can pass the remaining kwargs to the optimizer constructor.
    optimizer_type = optimizer_kwargs.pop("type", None)
    if optimizer_type is None:
        raise KeyError("Optimizer type needs to be specified!")

    # Get optimizer from optimizer type
    match optimizer_type.lower():
        case "adagrad":
            return torch.optim.Adagrad(model_parameters, **optimizer_kwargs)
        case "adam":
            return torch.optim.Adam(model_parameters, **optimizer_kwargs)
        case "adamw":
            return torch.optim.AdamW(model_parameters, **optimizer_kwargs)
        case "lbfgs":
            return torch.optim.LBFGS(model_parameters, **optimizer_kwargs)
        case "rmsprop":
            return torch.optim.RMSprop(model_parameters, **optimizer_kwargs)
        case "sgd":
            return torch.optim.SGD(model_parameters, **optimizer_kwargs)
        case _:
            raise ValueError(f"Invalid optimizer type: `{optimizer_type}`")


def get_scheduler_from_kwargs(
    optimizer: torch.optim.Optimizer,
    **scheduler_kwargs: Any,
) -> lrs.LRScheduler | lrs.ReduceLROnPlateau:
    """
    Builds and returns a learning rate scheduler for `optimizer`, based
    on the `scheduler_kwargs`.

    Args:
        optimizer: Optimizer for which the scheduler is used.
        **scheduler_kwargs: Keyword arguments for the scheduler, such
            at the patience or factor. Must also contain the key `type`,
            which specifies the scheduler type.

    Returns:
        Learning rate scheduler for the optimizer.
    """

    # Get scheduler type. We use `pop` to remove the key from the dictionary,
    # so that we can pass the remaining kwargs to the scheduler constructor.
    scheduler_type = scheduler_kwargs.pop("type")
    if scheduler_type is None:
        raise KeyError("Scheduler type needs to be specified!")

    # Get scheduler from scheduler type
    # Note: ReduceLROnPleateau is not a subclass of LRScheduler, so we need
    # to check for it separately.
    match scheduler_type.lower():
        case "step":
            return lrs.StepLR(optimizer, **scheduler_kwargs)
        case "cosine":
            return lrs.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        case "onecycle":
            return lrs.OneCycleLR(optimizer, **scheduler_kwargs)
        case "reduce_on_plateau":
            return lrs.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        case _:
            raise ValueError("No valid scheduler specified!")


def perform_scheduler_step(
    scheduler: lrs.LRScheduler | lrs.ReduceLROnPlateau,
    loss: Any = None,
    end_of: Literal["epoch", "batch"] = "epoch",
) -> None:
    """
    Wrapper for `scheduler.step()`. If scheduler is `ReduceLROnPlateau`,
    then `scheduler.step(loss)` is called, if not, `scheduler.step()`.

    Args:
        scheduler: Scheduler for learning rate.
        loss: Validation loss (only required for `ReduceLROnPlateau`).
        end_of: Whether the scheduler is called at the end of an epoch
            or at the end of a batch.
    """

    # Different schedulers need to be called at different times:
    if isinstance(scheduler, lrs.CosineAnnealingLR) and end_of == "epoch":
        scheduler.step()
    elif isinstance(scheduler, lrs.OneCycleLR) and end_of == "batch":
        scheduler.step()
    elif isinstance(scheduler, lrs.ReduceLROnPlateau) and end_of == "epoch":
        scheduler.step(loss)
    elif isinstance(scheduler, lrs.StepLR) and end_of == "epoch":
        scheduler.step()


def get_lr(optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Returns a list with the learning rates of the optimizer.
    """

    return [
        float(param_group["lr"])
        for param_group in optimizer.state_dict()["param_groups"]
    ]


def split_dataset_into_train_and_test(
    dataset: torch.utils.data.Dataset,
    train_fraction: float,
    random_seed: int = 42,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split the given `dataset` into a train set and a test set.

    The size of the train set is `int(train_fraction * len(dataset))`,
    the size of the test set is `len(dataset) - len(trainset)`.

    Args:
        dataset: Dataset to be split.
        train_fraction: Fraction of the dataset to be used for the
            train set. Must be in [0, 1].
        random_seed: Random seed for reproducibility.

    Returns:
        A 2-tuple: `(train_dataset, test_dataset)`.
    """

    generator = torch.Generator().manual_seed(random_seed)

    # The genereric `Dataset` class does not have a `_len__` method, so we
    # need to ignore the type warning here. Any actual dataset that is passed
    # in will have a `__len__` method, so this should be fine.
    # See: https://github.com/pytorch/pytorch/issues/25247
    train_size = int(train_fraction * len(dataset))  # type: ignore
    test_size = len(dataset) - train_size  # type: ignore

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[train_size, test_size],
        generator=generator,
    )

    return train_dataset, test_dataset


def build_train_and_test_loaders(
    dataset: torch.utils.data.Dataset,
    train_fraction: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool = True,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test `DataLoaders` for the given `dataset`.

    Args:
        dataset: Full dataset to be split into train and test sets.
        train_fraction: Fraction of the dataset to be used for the
            train set. Must be in [0, 1].
        batch_size: Batch size for the train and test loaders.
        num_workers: Number of workers for the train and test loaders.
        drop_last: Whether to drop the last batch if it is smaller than
            `batch_size`. This is only used for the train loader.
        random_seed: Random seed for reproducibility.

    Returns:
        A 2-tuple: `(train_loader, test_loader)`.
    """

    # Split the dataset into train and test sets
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        dataset=dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    # Build the corresponding DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(random_seed),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(random_seed),
    )

    return train_loader, test_loader


def validate_dims(x: torch.Tensor, ndim: int) -> None:
    """
    Validate that `x` has the correct number of dimensions.

    Raises:
        ValueError: If `x.ndim != ndim`.

    Args:
        x: A tensor.
        ndim: The expected number of dimensions.
    """

    # Use f-string hack to get the name of x
    name = f'{x=}'.split('=')[0].strip()

    if x.ndim != ndim:
        raise ValueError(
            f"Expected `{name}` to have {ndim} dimensions but found {x.ndim}!"
        )


def validate_shape(x: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    """
    Validate that `x` has the correct shape.

    Args:
        x: A tensor.
        shape: The expected shape. `None` means that the dimension can
            have any size.
    """

    # Use f-string hack to get the name of x
    name = f'{x=}'.split('=')[0].strip()

    # Check if the number of dimensions is correct
    if len(x.shape) != len(shape):
        raise ValueError(
            f"Expected `{name}` to have shape {shape} but found {x.shape}!"
        )

    # Check if the size of each dimension is correct
    for expected, actual in zip(shape, x.shape):
        if expected is not None and expected != actual:
            raise ValueError(
                f"Expected `{name}` to have shape {shape} but found {x.shape}!"
            )


def get_weights_from_checkpoint(
    file_path: Path,
    prefix: str,
) -> OrderedDict[str, torch.Tensor]:
    """
    Load the weights from a checkpoint file that starts with `prefix`.

    Args:
        file_path: Path to the checkpoint file.
        prefix: Prefix that the weights must start with. Usually, this
            is the name of a model component, e.g., `vectorfield_net`.

    Returns:
        An OrderecDict with the weights that can be loaded into a model.
    """

    # Load the full checkpoint
    checkpoint = torch.load(file_path)

    # Get the weights that start with `prefix`
    weights = OrderedDict(
        (key, value) for key, value in checkpoint["model_state_dict"].items()
        if key.startswith(prefix)
    )

    return weights


def load_and_or_freeze_model_weights(
    model: torch.nn.Module,
    freeze_weights: bool = False,
    load_weights: dict[str, str] | None = None,
) -> None:
    """
    Load and / or freeze weights of the given model, if requested.

    Args:
        model: The model to be modified.
        freeze_weights: Whether to freeze all weights of the model.
        load_weights: A dictionary with the following keys:
            - `file_path`: Path to the checkpoint file.
            - `prefix`: Prefix that the weights must start with.
                Usually, this is the name of a model component, e.g.,
                "vectorfield_net" or "context_embedding_net".
            If `None` or `{}` is passed, no weights are loaded.
    """

    # Load model weights from a file, if requested
    if load_weights is not None and load_weights:
        state_dict = get_weights_from_checkpoint(
            file_path=Path(load_weights["file_path"]),
            prefix=load_weights["prefix"],
        )
        model.load_state_dict(state_dict)

    # Freeze weights, if requested
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
