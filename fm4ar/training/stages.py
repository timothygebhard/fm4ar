"""
Methods for preparing a stage and running the training for it.
"""

from enum import Enum
from typing import Literal, TYPE_CHECKING
from zlib import adler32

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


class ExitStatus(str, Enum):
    """
    Exit status for running model.train() on a given stage.
    """

    COMPLETED = "COMPLETED"
    EARLY_STOPPED = "EARLY_STOPPED"
    MAX_RUNTIME_EXCEEDED = "MAX_RUNTIME_EXCEEDED"


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
    stage_name: str,
    stage_config: StageConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Initialize a training stage, that is, build the train and test
    loaders and initialize the optimizer and scheduler.

    Args:
        model: Instance of the model.
        dataset: Instance of the dataset.
        resume: Whether to resume from a checkpoint.
        stage_name: The name of the stage. This is only used to set
            the random seed for the data loaders to ensure that the
            batch order is not exactly the same for each stage.
        stage_config: The configuration for the stage.

    Returns:
        The `train_loader` and `valid_loader` for the stage.
    """

    # If we are not resuming, we should (re)set the stage_epoch and stage_name
    if not resume:
        model.stage_epoch = 0
        model.stage_name = stage_name

    # Construct stage-specific transforms for the dataset
    # These are the transforms that will be applied to the dataset in
    # __getitem__() and that are specific to the current stage (e.g.,
    # adding noise to the flux, re-binning the spectra, ...).
    dataset.data_transforms = get_data_transforms(stage_config.data_transforms)

    # Get the dataset configuration
    dataset_config = DatasetConfig(**model.config["dataset"])

    # Convert the stage name to an offset for the random seed that we can use
    # to ensure that the data loaders are not exactly the same for each stage.
    # The Adler-32 checksum is a simple and fast hash function that should be
    # good enough for this purpose. The bitwise AND operation ensures that the
    # offset is a positive integer in the range [0, 2^32 - 1].
    # [Note: the builtin hash() might seem simpler, but is not deterministic.]
    offset = adler32(stage_name.encode()) & 0xffffffff

    # Create the train and test data loaders
    # Allows changes in batch size between stages
    train_loader, test_loader = build_dataloaders(
        dataset=dataset,
        n_train_samples=dataset_config.n_train_samples,
        n_valid_samples=dataset_config.n_valid_samples,
        batch_size=stage_config.batch_size,
        n_workers=get_number_of_workers(stage_config.n_workers),
        drop_last=stage_config.drop_last,
        random_seed=dataset_config.random_seed + offset,
    )

    # Create a new optimizer and scheduler
    # If we are resuming, these should have been loaded from the checkpoint
    if not resume:
        print("Initializing new optimizer and scheduler!")
        model.optimizer_config = stage_config.optimizer
        model.scheduler_config = stage_config.scheduler
        model.initialize_optimizer_and_scheduler()

    # Set the precision for float32 matrix multiplication (globally)
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

    # Initialize the runtime limits
    runtime_limits = RuntimeLimits(
        max_runtime=model.config["local"].get("max_runtime", None),
    )

    # Find the index of the first stage that we haven't completed and get
    # the stage configurations for this and all subsequent stages. Keep in
    # mind that stages can use early stopping, so looking at `model.epoch`
    # is not sufficient and we need to track the state more explicitly.
    stage_names = list(model.config["training"].keys())
    start_idx = (
        0 if model.stage_name is None
        else stage_names.index(model.stage_name)
    )
    stage_names_and_configs = {
        name: StageConfig(**stage)
        for name, stage in list(model.config["training"].items())[start_idx:]
    }

    # Check for the first stage in the loop below if we are resuming.
    # Usually, this should always be True, except if we start a completely
    # new training run. In all other cases, we should have trained at least
    # one epoch per stage (unless we time out while saving the model).
    resume = model.stage_epoch > 0

    # Initialize exit status
    exit_status = ExitStatus.COMPLETED

    # Loop over the remaining stages
    for stage_name, stage_config in stage_names_and_configs.items():

        # Initialize the stage (either from scratch or from a checkpoint)
        begin_or_resume = "Resuming" if resume else "Beginning"
        print(f"\n{begin_or_resume} training stage '{stage_name}'")

        # Initialize the train and test loaders for the stage
        train_loader, test_loader = initialize_stage(
            model=model,
            dataset=dataset,
            resume=resume,
            stage_name=stage_name,
            stage_config=stage_config,
        )

        # (Re)set runtime limit for the stage
        runtime_limits.max_epochs = stage_config.epochs

        # Train the model for the stage
        exit_status = model.train(
            train_loader=train_loader,
            valid_loader=test_loader,
            runtime_limits=runtime_limits,
            stage_config=stage_config,
        )

        # Save the model if we have reached the end of the stage
        if exit_status == ExitStatus.COMPLETED:
            print(f"Training stage '{stage_name}' complete!")
            print("Saving model...", end=" ")
            model.save_model(name=stage_name, save_training_info=True)
            print("Done!\n", flush=True)

        # If we start another loop, we will definitely not be resuming but
        # instead be starting a new stage from scratch
        resume = False

    # Check if we have reached the end of the training, either because we
    # stopped early or because we have completed all the stages
    return exit_status in [ExitStatus.COMPLETED, ExitStatus.EARLY_STOPPED]
