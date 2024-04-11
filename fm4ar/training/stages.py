"""
Methods for preparing a stage and running the training for it.
"""

from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from fm4ar.datasets import DatasetConfig
from fm4ar.datasets.dataset import SpectraDataset
from fm4ar.datasets.data_transforms import (
    DataTransformConfig,
    get_data_transforms,
)
from fm4ar.torchutils.dataloaders import (
    build_dataloaders,
    get_number_of_workers,
)
from fm4ar.torchutils.gradient_clipping import GradientClippingConfig
from fm4ar.torchutils.optimizers import OptimizerConfig
from fm4ar.torchutils.schedulers import SchedulerConfig
from fm4ar.utils.tracking import RuntimeLimits

if TYPE_CHECKING:  # pragma: no cover
    from fm4ar.models.base import Base


class StageConfig(BaseModel):
    """
    Configuration for a training stage.
    """

    batch_size: int = Field(
        ...,
        description="Batch size for the stage.",
    )
    data_transforms: list[DataTransformConfig] = Field(
        ...,
        description="Transforms to apply to the dataset (e.g, add noise).",
    )
    drop_last: bool = Field(
        default=True,
        description=(
            "Whether the training dataloader should drop the last batch "
            "if it is smaller than the batch size."
        ),
    )
    early_stopping: int = Field(
        ...,
        description="Number of epochs without improvement before stopping.",
    )
    epochs: int = Field(
        ...,
        description="Number of epochs to train the model for.",
    )
    float32_matmul_precision: str = Field(
        default="highest",
        description="Precision for float32 matrix multiplication.",
    )
    gradient_clipping: GradientClippingConfig
    logprob_epochs: int | None = Field(
        ...,
        description="Number of epochs between log-probability calculation.",
    )
    loss_kwargs: dict = Field(
        default={},
        description=(
            "Additional keyword arguments for the loss function. This can "
            "be used, e.g., to control the `time_prior_exponent` per stage."
        ),
    )
    n_workers: int | Literal["auto"] = Field(
        default="auto",
        description="Number of workers for the data loaders.",
    )
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    use_amp: bool = Field(
        ...,
        description="Whether to use automatic mixed precision training.",
    )


def initialize_stage(
    model: "Base",
    dataset: SpectraDataset,
    resume: bool,
    stage_number: int,
    stage_config: StageConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Initialize a training stage, that is, build the train and test
    loaders and initialize the optimizer and scheduler.

    Args:
        model: Instance of the model.
        dataset: Instance of the dataset.
        resume: Whether to resume from a checkpoint.
        stage_number: The number of the stage. This is only used to
            set the random seed for the data loaders to ensure that
            the batch order is not exactly the same for each stage.
        stage_config: The configuration for the stage.

    Returns:
        The `train_loader` and `valid_loader` for the stage.
    """

    # Construct stage-specific transforms for the dataset
    # These are the transforms that will be applied to the dataset in
    # __getitem__() and that are specific to the current stage (e.g.,
    # adding noise to the flux, re-binning the spectra, ...).
    dataset.data_transforms = get_data_transforms(stage_config.data_transforms)

    # Get the dataset configuration
    dataset_config = DatasetConfig(**model.config["dataset"])

    # Create the train and test data loaders
    # Allows changes in batch size between stages
    train_loader, test_loader = build_dataloaders(
        dataset=dataset,
        n_train_samples=dataset_config.n_train_samples,
        n_valid_samples=dataset_config.n_valid_samples,
        batch_size=stage_config.batch_size,
        n_workers=get_number_of_workers(stage_config.n_workers),
        drop_last=stage_config.drop_last,
        random_seed=dataset_config.random_seed * (stage_number + 1),
    )

    # Create a new optimizer and scheduler
    # If we are resuming, these should have been loaded from the checkpoint
    if not resume:
        print("Initializing new optimizer and scheduler!")
        model.optimizer_config = stage_config.optimizer
        model.scheduler_config = stage_config.scheduler
        model.initialize_optimizer_and_scheduler()

    # Set the precision for float32 matrix multiplication
    torch.set_float32_matmul_precision(stage_config.float32_matmul_precision)

    return train_loader, test_loader


def train_stages(
    model: "Base",
    dataset: SpectraDataset,
) -> bool:
    """
    Train the network, iterating through the sequence of stages.
    Stages can change certain settings (e.g., for the optimizer).

    Args:
        model: Model.
        dataset: Dataset (before train / validation split).

    Returns:
        A boolean: `True` if all stages are complete, `False` otherwise.
    """

    # Initialize the runtime limits (e.g., max number of epochs)
    runtime_limits = RuntimeLimits(
        epoch_start=model.epoch,
        **model.config["local"]["runtime_limits"],
    )

    # Initialize the flag that indicates whether we stopped early
    stopped_early = False

    # Extract list of stages from settings dict
    stage_names = list(model.config["training"].keys())
    stage_configs = [
        StageConfig(**stage_config_dict)
        for stage_config_dict in model.config["training"].values()
    ]
    num_stages = len(stage_configs)

    # Get the total number of epochs at the end of each stage (cumulative sum),
    # and use it to determine the stage in which we are starting
    end_epochs = list(np.cumsum([stage.epochs for stage in stage_configs]))
    num_starting_stage = np.searchsorted(end_epochs, model.epoch + 1)

    # Find the starting stage and iterate over the remaining stages
    for stage_number in range(num_starting_stage, num_stages):

        # Get the name, configuration and overall starting epoch for the stage
        stage_name = stage_names[stage_number]
        stage_config = stage_configs[stage_number]
        stage_start_epoch = end_epochs[stage_number] - stage_config.epochs

        # Initialize the stage (either from scratch or from a checkpoint)
        resume = bool(model.epoch > stage_start_epoch)
        begin_or_resume = "Resuming" if resume else "Beginning"
        print(f"\n{begin_or_resume} training stage '{stage_name}'")

        # Initialize the train and test loaders for the stage
        train_loader, test_loader = initialize_stage(
            model=model,
            dataset=dataset,
            resume=resume,
            stage_config=stage_config,
            stage_number=stage_number,
        )

        # Update the runtime limits for the stage
        runtime_limits.max_epochs_total = end_epochs[stage_number]

        # Train the model for the stage
        stopped_early = model.train(
            train_loader=train_loader,
            valid_loader=test_loader,
            runtime_limits=runtime_limits,
            stage_config=stage_config,
        )

        # Save the model if we have reached the end of the stage
        if model.epoch == end_epochs[stage_number]:
            print(f"Training stage '{stage_name}' complete!")
            print("Saving model...", end=" ")
            model.save_model(name=stage_name, save_training_info=True)
            print("Done!\n", flush=True)

    # Check if we have reached the end of the training, either because we
    # stopped early or because we have completed all stages
    complete = stopped_early or model.epoch == end_epochs[-1]
    return complete
