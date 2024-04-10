"""
Methods for preparing a stage and running the training for it.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from fm4ar.datasets.dataset import SpectraDataset
from fm4ar.datasets.data_transforms import get_data_transforms
from fm4ar.models.base import Base
from fm4ar.utils.torchutils import (
    build_train_and_test_loaders,
    get_number_of_workers,
)
from fm4ar.utils.tracking import RuntimeLimits


def initialize_stage(
    model: Base,
    dataset: SpectraDataset,
    n_workers: int,
    resume: bool,
    stage_config: dict,
    stage_number: int,
) -> tuple[DataLoader, DataLoader]:
    """
    Initialize a training stage, that is, build the train and test
    loaders and initialize the optimizer and scheduler.

    Args:
        model: Model.
        dataset: Dataset.
        stage_config: Dictionary containing the config for a stage.
        stage_number: The number of the stage. This is only used to
            set the random seed for the data loaders to ensure that
            the batch order is not exactly the same for each stage.
        n_workers: Number of workers for the data loaders.
        resume: Whether to resume from a checkpoint.

    Returns:
        The `train_loader` and `test_loader` for the stage.
    """

    # Construct stage-specific transforms for the dataset
    # These are the transforms that will be applied to the dataset in
    # __getitem__() and that are specific to the current stage (e.g.,
    # adding noise to the flux, re-binning the spectra, ...).
    dataset.data_transforms = get_data_transforms(stage_config)

    # Create the train and test data loaders
    # Allows changes in batch size between stages
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dataset,
        train_fraction=model.config["dataset"]["train_fraction"],
        batch_size=stage_config["batch_size"],
        num_workers=n_workers,
        drop_last=stage_config.get("drop_last", True),
        random_seed=stage_number,
    )

    # Create a new optimizer and scheduler
    # If we are resuming, these should have been loaded from the checkpoint
    if not resume:
        print("Initializing new optimizer and scheduler!")
        model.optimizer_config = stage_config["optimizer"]
        model.scheduler_config = stage_config["scheduler"]
        model.initialize_optimizer_and_scheduler()

    # Set the precision for fp32 matrix multiplication
    precision = stage_config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)

    return train_loader, test_loader


def train_stages(
    model: Base,
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
    stage_configs = list(model.config["training"].values())
    num_stages = len(stage_configs)

    # Get the total number of epochs at the end of each stage (cumulative sum),
    # and use it to determine the stage in which we are starting
    end_epochs = list(np.cumsum([stage["epochs"] for stage in stage_configs]))
    num_starting_stage = np.searchsorted(end_epochs, model.epoch + 1)

    # Find the starting stage and iterate over the remaining stages
    for stage_number in range(num_starting_stage, num_stages):

        # Get the name, configuration and overall starting epoch for the stage
        stage_name = list(model.config["training"].keys())[stage_number]
        stage_config = stage_configs[stage_number]
        stage_start_epoch = end_epochs[stage_number] - stage_config["epochs"]

        # Initialize the stage (either from scratch or from a checkpoint)
        resume = bool(model.epoch > stage_start_epoch)
        begin_or_resume = "Resuming" if resume else "Beginning"
        print(f"\n{begin_or_resume} training stage '{stage_name}'")

        # Determine the number of workers for the data loaders
        # There's usually no need to change this, but it can be set in the
        # stage configuration if desired. "auto" means all cores - 1.
        n_workers = get_number_of_workers(
            stage_config.get("n_workers", "auto")
        )

        # Initialize the train and test loaders for the stage
        train_loader, test_loader = initialize_stage(
            model=model,
            dataset=dataset,
            n_workers=n_workers,
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
