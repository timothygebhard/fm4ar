"""
Utility functions for PyTorch.
"""

import platform
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from importlib import import_module
from math import prod
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Type

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.optim import lr_scheduler as lrs
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from fm4ar.nn.modules import Sine
from fm4ar.utils.multiproc import get_number_of_available_cores


def build_train_and_test_loaders(
    dataset: Dataset,
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

    # Define the worker init function: This will set the random seed for the
    # workers. See documentation of `manual_seed_without_return` for details.
    worker_init_fn = partial(
        manual_seed_without_return,
        random_seed=random_seed,
    )

    # Build the train loader
    torch.manual_seed(random_seed)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    # Build the test loader
    torch.manual_seed(random_seed + 1)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, test_loader


def check_for_nans(x: torch.Tensor, label: str = "tensor") -> None:
    """
    Check if the given tensor (usually the loss) contains any entries
    that are NaN or infinite. If so, raise a `ValueError`.
    """

    if torch.isnan(x).any():
        raise ValueError(f"NaN values detected in {label}, aborting!")
    if torch.isinf(x).any():
        raise ValueError(f"Inf values detected in {label}, aborting!")


def get_activation_from_name(name: str) -> torch.nn.Module:
    """
    Build and return an activation function with the given name.
    """

    ActivationFunction: Type[torch.nn.Module]

    # Custom activation functions need special treatment
    if name == "Sine":
        ActivationFunction = Sine

    # All other activation functions are built from `torch.nn`
    else:
        try:
            ActivationFunction = getattr(import_module("torch.nn"), name)
        except AttributeError as e:
            raise ValueError(f"Invalid activation function: `{name}`") from e

    # Instantiate the activation function and return it
    return ActivationFunction()


def get_lr(optimizer: torch.optim.Optimizer) -> list[float]:
    """
    Returns a list with the learning rates of the optimizer.
    """

    return [
        float(param_group["lr"])
        for param_group in optimizer.state_dict()["param_groups"]
    ]


def get_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
    batch_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build and return an MLP with the given parameters.

    Args:
        input_dim: Dimension of the input.
        hidden_dims: List of hidden dimensions.
        output_dim: Dimension of the output.
        activation: Name of the activation function.
        batch_norm: Whether to use batch normalization.
        dropout: Dropout probability (between 0 and 1).

    Returns:
        A multi-layer perceptron with the given parameters.
    """

    # Prepare list of layers
    layers = torch.nn.ModuleList()
    dims = [input_dim] + list(hidden_dims) + [output_dim]

    # Note: There seems to be no clear consensus about the order of the
    # activation function and the batch normalization layer.
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(get_activation_from_name(activation))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))

    return nn.Sequential(*layers)


def get_number_of_parameters(
    model: nn.Module,
    requires_grad_flags: tuple[bool, ...] = (True, False),
) -> int:
    """
    Count the number of parameters of the given `model`.

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


def get_number_of_workers(n_workers: int | Literal["auto"]) -> int:
    """
    Determine the number of workers for a `DataLoader`.

    Args:
        n_workers: If an integer is given, this is returned. If "auto"
            is given, we determine the number of cores based on the
            host system.

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
    n_available_cores = get_number_of_available_cores()
    return max(n_available_cores - 1, 1)


def get_optimizer_from_config(
    model_parameters: Iterable,
    optimizer_config: dict,
) -> torch.optim.Optimizer:
    """
    Builds and returns an optimizer for `model_parameters`, based on
    the given `optimizer_config`.

    This should support all optimizers from `torch.optim`.
    Note: Some optimizers, such as LBFGS, can be specicied like this
    but will not work for training as they require the `closure` to be
    defined (see the PyTorch documentation for details).

    Args:
        model_parameters: Model parameters to optimize.
        optimizer_config: Configuration for the optimizer. Must contain
        the keys `type` (e.g., "Adam") and `kwargs` (e.g, with `lr`).

    Returns:
        Optimizer for the model parameters.
    """

    # Get optimizer type and additional keyword arguments from config
    optimizer_type = optimizer_config["type"]
    optimizer_kwargs = optimizer_config.get("kwargs", {})

    # Get optimizer class based on the type
    try:
        OptimizerClass: Type[torch.optim.Optimizer]
        OptimizerClass = getattr(
            import_module("torch.optim"),
            optimizer_type,
        )
    except AttributeError as e:
        raise ValueError(f"Invalid optimizer type: `{optimizer_type}`") from e

    # Instantiate optimizer from optimizer type and keyword arguments
    return OptimizerClass(model_parameters, **optimizer_kwargs)


def get_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict,
) -> lrs.LRScheduler | lrs.ReduceLROnPlateau:
    """
    Builds and returns a learning rate scheduler for `optimizer`, based
    on the given `scheduler_config`.

    This should support all schedulers from `torch.optim.lr_scheduler`.

    Args:
        optimizer: Optimizer for which the scheduler is used.
        scheduler_config: Configuration for the scheduler. Must contain
            the key `type` (e.g., "ReduceLROnPlateau") and `kwargs`.

    Returns:
        Learning rate scheduler for the optimizer.
    """

    # Get scheduler type and additional keyword arguments
    scheduler_type = scheduler_config["type"]
    scheduler_kwargs = scheduler_config.get("kwargs", {})

    # Get scheduler class based on the type
    try:
        SchedulerClass: Type[lrs.LRScheduler | lrs.ReduceLROnPlateau]
        SchedulerClass = getattr(
            import_module("torch.optim.lr_scheduler"),
            scheduler_type,
        )
    except AttributeError as e:
        raise ValueError(f"Invalid scheduler type: `{scheduler_type}`") from e

    # Instantiate scheduler from scheduler type and keyword arguments
    return SchedulerClass(optimizer, **scheduler_kwargs)


def get_weights_from_pt_file(
    file_path: Path,
    state_dict_key: str,
    prefix: str,
    drop_prefix: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """
    Load the weights that starts with `prefix` from a *.pt file.

    Args:
        file_path: Path to the *.pt file.
        state_dict_key: Key of the state dict in the *.pt file that
            contains the weights. Usually, this is "model_state_dict".
        prefix: Prefix that the weights must start with. Usually, this
            is the name of a model component, e.g., `vectorfield_net`.
        drop_prefix: Whether to drop the prefix from the keys of the
            returned dictionary.

    Returns:
        An OrderecDict with the weights that can be loaded into a model.
    """

    # Load the full checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(file_path, map_location=device)

    # Select the state dict that contains the weights
    state_dict = checkpoint[state_dict_key]

    # Get the weights that start with `prefix`
    weights = OrderedDict(
        (key if not drop_prefix else key.removeprefix(prefix + "."), value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )

    return weights


def load_and_or_freeze_model_weights(
    model: torch.nn.Module,
    freeze_weights: bool = False,
    load_weights: dict | None = None,
) -> None:
    """
    Load and / or freeze weights of the given model, if requested.

    Args:
        model: The model to be modified.
        freeze_weights: Whether to freeze all weights of the model.
        load_weights: A dictionary with the following keys:
            - `file_path`: Path to the checkpoint file (`*.pt`).
            - `state_dict_key`: Key of the state dict in the checkpoint
                file that contains the weights. Usually, this is
                "model_state_dict".
            - `prefix`: Prefix that the weights must start with.
                Usually, this is the name of a model component, e.g.,
                "vectorfield_net" or "context_embedding_net".
            - `drop_prefix`: Whether to drop the prefix from the keys.
                Default is `True`.
            If `None` or `{}` is passed, no weights are loaded.
    """

    # Load weights, if requested
    if load_weights is not None and load_weights:

        # Validator for the `load_weights` dictionary
        # Seems cleaner than a lot of `if` statements and ValueErrors?
        class LoadWeightsConfig(BaseModel):
            file_path: Path
            state_dict_key: str
            prefix: str
            drop_prefix: bool = True

        # Validate the `load_weights` dictionary
        load_weights_config = LoadWeightsConfig(**load_weights)

        # Load model weights from a file, if requested
        state_dict = get_weights_from_pt_file(
            file_path=load_weights_config.file_path,
            state_dict_key=load_weights_config.state_dict_key,
            prefix=load_weights_config.prefix,
            drop_prefix=load_weights_config.drop_prefix,
        )
        model.load_state_dict(state_dict)

    # Freeze weights, if requested
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False


def manual_seed_without_return(
    worker_id: int,  # Note: The argument order matters here!
    random_seed: int,
) -> None:
    """
    Minimal wrapper around `torch.manual_seed()` that returns None.

    This is used as the `worker_init_fn` for the `DataLoader` to set the
    random seed for the workers. A simpler solution could have been to
    use `lambda worker_id: torch.manual_seed(worker_id + random_seed)`,
    but the `worker_init_fn` is supposed to return None, hence this
    workaround.
    """

    torch.manual_seed(random_seed + worker_id)
    return None


def perform_scheduler_step(
    scheduler: lrs.LRScheduler | lrs.ReduceLROnPlateau,
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


def split_dataset_into_train_and_test(
    dataset: Dataset,
    train_fraction: float,
    random_seed: int = 42,
) -> tuple[Subset, Subset]:
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
        A 2-tuple `(train_dataset, test_dataset)`, which are disjoint
        subsets of the original `dataset`.
    """

    generator = torch.Generator().manual_seed(random_seed)

    # The genereric `Dataset` class does not have a `_len__` method, so we
    # need to ignore the type warning here. Any actual dataset that is passed
    # in will have a `__len__` method, so this should be fine.
    # See: https://github.com/pytorch/pytorch/issues/25247
    train_size = int(train_fraction * len(dataset))  # type: ignore
    test_size = len(dataset) - train_size  # type: ignore

    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, test_size],
        generator=generator,
    )

    return train_dataset, test_dataset
