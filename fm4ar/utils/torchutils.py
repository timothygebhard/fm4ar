"""
Utility functions for PyTorch.
"""

from collections import OrderedDict
from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import Any, Iterable, Literal, Type

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler as lrs
from torch.utils.data import DataLoader, Dataset, random_split

from fm4ar.nn.modules import Sine
from fm4ar.utils.resampling import resample_spectrum


def check_for_nans(x: torch.Tensor, label: str = "tensor") -> None:
    """
    Check if the given tensor (usually the loss) contains any entries
    that are NaN or infinite. If so, raise a `ValueError`.
    """

    if torch.isnan(x).any():
        raise ValueError(f"NaN values detected in {label}, aborting!")
    if torch.isinf(x).any():
        raise ValueError(f"Inf values detected in {label}, aborting!")


def get_activation_from_string(name: str) -> torch.nn.Module:
    """
    Build and return an activation function with the given name.
    """

    match name.lower():
        case "elu":
            return torch.nn.ELU()
        case "gelu":
            return torch.nn.GELU()
        case "leaky_relu":
            return torch.nn.LeakyReLU()
        case "relu":
            return torch.nn.ReLU()
        case "sigmoid":
            return torch.nn.Sigmoid()
        case "sine":
            return Sine()
        case "swish":
            return torch.nn.SiLU()
        case "tanh":
            return torch.nn.Tanh()
        case _:
            raise ValueError("Invalid activation function!")


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
            layers.append(get_activation_from_string(activation))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))

    return nn.Sequential(*layers)


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

    # Map strings to scheduler classes
    mapping: dict[str, Type[lrs.LRScheduler | lrs.ReduceLROnPlateau]] = {
        "step": lrs.StepLR,
        "cosine": lrs.CosineAnnealingLR,
        "cosine_warm_restarts": lrs.CosineAnnealingWarmRestarts,
        "cyclic": lrs.CyclicLR,
        "onecycle": lrs.OneCycleLR,
        "reduce_on_plateau": lrs.ReduceLROnPlateau,
    }

    # Get scheduler from scheduler type
    try:
        return mapping[scheduler_type.lower()](optimizer, **scheduler_kwargs)
    except KeyError as e:
        raise ValueError(f"Invalid scheduler type: `{scheduler_type}`") from e


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
    if (
        isinstance(scheduler, lrs.CosineAnnealingWarmRestarts)
        and end_of == "epoch"
    ):
        scheduler.step()
    if isinstance(scheduler, lrs.CyclicLR) and end_of == "batch":
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
    dataset: Dataset,
    train_fraction: float,
    random_seed: int = 42,
) -> tuple[Dataset, Dataset]:
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

    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, test_size],
        generator=generator,
    )

    return train_dataset, test_dataset


def collate_and_corrupt(
    batch_as_list: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate the given batch and corrupt it by resampling the wavelength
    grid and randomly selecting a subset of wavelengths.

    This function can be passed as a `collate_fn` to a `DataLoader`.

    Args:
        batch_as_list: Batch as a list of tuples `(theta_i, x_i)`.

    Returns:
        A 2-tuple: `(theta, x)`.
    """

    # Define some constants that might require tweaking
    RESAMPLING_STD = 0.1
    DISCARD_FRACTION = 0.2  # discard 20% of the wavelengths

    # Determine target wavelength grid for resampling
    # Note: Because the first and last bin usually end up containing NaNs,
    # this is different from the `new_wlen` below which is the actual grid.
    resampling_factor = float(10 ** np.random.normal(0, RESAMPLING_STD))
    n_bins_original = int(batch_as_list[0][1].shape[0])
    min_wlen = float(batch_as_list[0][1][:, 1].min())
    max_wlen = float(batch_as_list[0][1][:, 1].max())
    n_bins_resampled = int(n_bins_original * resampling_factor)
    target_wlen = np.linspace(min_wlen, max_wlen, n_bins_resampled)

    # Create the "normal" batch
    theta_list, x_list = [], []
    for theta_i, x_i in batch_as_list:
        theta_list.append(theta_i)
        x_list.append(x_i)
    theta = torch.stack(theta_list)

    # Resample to a different wavelength grid
    resampled_x_list = []
    for x_i in x_list:
        old_flux, old_wlen, old_errs = x_i[:, 0], x_i[:, 1], x_i[:, 2]
        new_wlen, new_flux, new_errs = resample_spectrum(
            new_wlen=target_wlen,
            old_wlen=old_wlen.numpy(),
            old_flux=old_flux.numpy(),
            old_errs=old_errs.numpy(),
        )
        resampled = torch.stack(
            [
                torch.from_numpy(new_flux),
                torch.from_numpy(new_wlen),
                torch.from_numpy(new_errs),
            ],
            dim=1,
        )
        resampled_x_list.append(resampled)
    x = torch.stack(resampled_x_list, dim=0)

    # Randomly select subset of wavelengths
    idx = torch.randperm(x.shape[1])
    mask = idx < (DISCARD_FRACTION * x.shape[1])
    x = x[:, mask].reshape(x.shape[0], -1, x.shape[2])

    return theta.float(), x.float()


def collate_pretrain(
    batch_as_list: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function should a list of the "vanilla" batches: original full
    resolution and no noise added yet.

    We now need to do the following:
      - Add some noise to the fluxes (we can do this because the last
        dimension of the context contains the noise levels)
      - Resample to a random wavelength grid
      - Randomly select a subset of wavelengths

    We then return *both* the original batch and the corrupted batch.

    Note: We can throw out the parameters `theta` here, because we will
    not need them for pre-training (which is all about learning to
    understand the structure of the spectra).
    """

    # Define some constants that might require tweaking
    RESAMPLING_STD = 0.1
    DISCARD_FRACTION = 0.5  # discard 50% of the wavelengths

    # Determine target wavelength grid for resampling
    # Note: Because the first and last bin usually end up containing NaNs,
    # this is different from the `new_wlen` below which is the actual grid.
    resampling_factor = float(10 ** np.random.normal(0, RESAMPLING_STD))
    n_bins_original = int(batch_as_list[0][1].shape[0])
    min_wlen = float(batch_as_list[0][1][:, 1].min())
    max_wlen = float(batch_as_list[0][1][:, 1].max())
    n_bins_resampled = int(n_bins_original * resampling_factor)
    target_wlen = np.linspace(min_wlen, max_wlen, n_bins_resampled)

    # Construct the "vanilla" batch (spectra without corruptions)
    # Note: This is where we drop theta
    x_list = []
    for _, x_i in batch_as_list:
        x_list.append(x_i)
    x_vanilla = torch.stack(x_list)

    # Add some noise to the fluxes
    for x_i in x_list:
        x_i[:, 0] += torch.randn_like(x_i[:, 2]) * x_i[:, 2]

    # Resample to a different wavelength grid
    for i in range(len(x_list)):
        x_i = x_list[i]
        old_flux, old_wlen, old_errs = x_i[:, 0], x_i[:, 1], x_i[:, 2]
        new_wlen, new_flux, new_errs = resample_spectrum(
            new_wlen=target_wlen,
            old_wlen=old_wlen.numpy(),
            old_flux=old_flux.numpy(),
            old_errs=old_errs.numpy(),
        )
        x_list[i] = torch.stack(
            [
                torch.from_numpy(new_flux),
                torch.from_numpy(new_wlen),
                torch.from_numpy(new_errs),
            ],
            dim=1,
        )

    # Stack noisy, resampled spectra into a tensor
    x_corrupted = torch.stack(x_list)

    # Randomly select subset of wavelengths
    idx = torch.randperm(x_corrupted.shape[1])
    mask = idx > (DISCARD_FRACTION * x_corrupted.shape[1])
    x_corrupted = x_corrupted[:, mask].reshape(
        x_corrupted.shape[0], -1, x_corrupted.shape[2]
    )

    return x_vanilla.float(), x_corrupted.float()


def build_train_and_test_loaders(
    dataset: Dataset,
    train_fraction: float,
    batch_size: int,
    num_workers: int,
    drop_last: bool = True,
    random_seed: int = 42,
    train_collate_fn: str | None = None,
    test_collate_fn: str | None = None,
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
        train_collate_fn: Name of collate function that is passed to the
            train loader. If `None`, use the default collate function.
        test_collate_fn: Name of collate function that is passed to the
            test loader. If `None`, use the default collate function.

    Returns:
        A 2-tuple: `(train_loader, test_loader)`.
    """

    # Split the dataset into train and test sets
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        dataset=dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )

    # Define collate functions
    collate_functions = {
        "collate_and_corrupt": collate_and_corrupt,
        "collate_pretrain": collate_pretrain,
        None: None,
    }

    # Build the corresponding DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(random_seed),
        collate_fn=collate_functions.get(train_collate_fn),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(random_seed),
        collate_fn=collate_functions.get(test_collate_fn),
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
    for expected, actual in zip(shape, x.shape, strict=True):
        if expected is not None and expected != actual:
            raise ValueError(
                f"Expected `{name}` to have shape {shape} but found {x.shape}!"
            )


def get_weights_from_pt_file(
    file_path: Path,
    prefix: str,
    drop_prefix: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """
    Load the weights that starts with `prefix` from a *.pt file.

    Args:
        file_path: Path to the *.pt file.
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

    # Get the weights that start with `prefix`
    weights = OrderedDict(
        (
            key if not drop_prefix else key.removeprefix(prefix + '.'),
            value
        ) for key, value in checkpoint.items() if key.startswith(prefix)
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
            - `file_path`: Path to the checkpoint file (`*.pt`).
            - `prefix`: Prefix that the weights must start with.
                Usually, this is the name of a model component, e.g.,
                "vectorfield_net" or "context_embedding_net".
            - `drop_prefix`: Whether to drop the prefix from the keys.
                Default is `True`.
            If `None` or `{}` is passed, no weights are loaded.
    """

    # Load model weights from a file, if requested
    if load_weights is not None and load_weights:
        state_dict = get_weights_from_pt_file(
            file_path=Path(load_weights["file_path"]),
            prefix=load_weights["prefix"],
            drop_prefix=bool(load_weights.get("drop_prefix", True)),
        )
        model.load_state_dict(state_dict)

    # Freeze weights, if requested
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False
