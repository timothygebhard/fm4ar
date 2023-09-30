"""
Methods for preparing a stage and running the training for it.
"""

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from fm4ar.datasets.dataset import ArDataset
from fm4ar.models.base import Base
from fm4ar.utils.torchutils import build_train_and_test_loaders
from fm4ar.utils.tracking import RuntimeLimits


def initialize_stage(
    pm: Base,
    dataset: ArDataset,
    stage_config: dict,
    num_workers: int,
    resume: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Initialize a training stage, that is, build the train and test
    loaders and initialize the optimizer and scheduler.

    Args:
        pm: Posterior model.
        dataset: Dataset.
        stage_config: Dictionary containing the config for a stage.
        num_workers: Number of workers for the data loaders.
        resume: Whether to resume from a checkpoint.

    Returns:
        The `train_loader` and `test_loader` for the stage.
    """

    # Create the train and test data loaders
    # Allows changes in batch size between stages
    train_loader, test_loader = build_train_and_test_loaders(
        dataset=dataset,
        train_fraction=pm.config["data"]["train_fraction"],
        batch_size=stage_config["batch_size"],
        num_workers=num_workers,
        drop_last=stage_config.get("drop_last", True),
        train_collate_fn=stage_config.get("train_collate_fn", None),
        test_collate_fn=stage_config.get("test_collate_fn", None),
    )

    # Create a new optimizer and scheduler
    # If we are resuming, these should have been loaded from the checkpoint
    if not resume:
        print("Initializing new optimizer and scheduler!")
        pm.optimizer_kwargs = stage_config["optimizer"]
        pm.scheduler_kwargs = stage_config["scheduler"]
        pm.initialize_optimizer_and_scheduler()

    # Set the precision for fp32 matrix multiplication
    precision = stage_config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)  # type: ignore

    return train_loader, test_loader


def train_stages(
    pm: Base,
    dataset: ArDataset,
) -> bool:
    """
    Train the network, iterating through the sequence of stages.
    Stages can change certain settings (e.g., for the optimizer).

    Args:
        pm: Posterior model.
        dataset: Dataset.

    Returns:
        A boolean: `True` if all stages are complete, `False` otherwise.
    """

    # Initialize the runtime limits (e.g., max number of epochs)
    runtime_limits = RuntimeLimits(
        epoch_start=pm.epoch,
        **pm.config["local"]["runtime_limits"],
    )

    # Extract list of stages from settings dict
    stage_configs = list(pm.config["training"].values())
    num_stages = len(stage_configs)

    # Get the total number of epochs at the end of each stage (cumulative sum),
    # and use it to determine the stage in which we are starting
    end_epochs = list(np.cumsum([stage["epochs"] for stage in stage_configs]))
    num_starting_stage = np.searchsorted(end_epochs, pm.epoch + 1)

    # Find the starting stage and iterate over the remaining stages
    for n in range(num_starting_stage, num_stages):

        # Get the name, configuration and overall starting epoch for the stage
        stage_name = list(pm.config["training"].keys())[n]
        stage_config = stage_configs[n]
        stage_start_epoch = end_epochs[n] - stage_config["epochs"]

        # Initialize the stage (either from scratch or from a checkpoint)
        resume = bool(pm.epoch > stage_start_epoch)
        begin_or_resume = "Resuming" if resume else "Beginning"
        print(f"\n{begin_or_resume} training stage '{stage_name}'")

        # Initialize the train and test loaders for the stage
        train_loader, test_loader = initialize_stage(
            pm=pm,
            dataset=dataset,
            stage_config=stage_config,
            num_workers=pm.config["local"]["num_workers"],
            resume=resume,
        )

        # Update the runtime limits for the stage
        runtime_limits.max_epochs_total = end_epochs[n]

        # Train the model for the stage
        pm.train(
            train_loader=train_loader,
            test_loader=test_loader,
            runtime_limits=runtime_limits,
            stage_config=stage_config,
        )

        # Save the model if we have reached the end of the stage
        if pm.epoch == end_epochs[n]:
            print(f"Training stage '{stage_name}' complete!")
            print("Saving model...", end=" ")
            pm.save_model(name=stage_name, save_training_info=True)
            print("Done!\n", flush=True)

        # Check if we have reached the runtime limits
        if runtime_limits.local_limits_exceeded(pm.epoch):
            print("Local runtime limits reached. Ending program.\n")
            break

    # Mark the run as finished on wandb
    # This should happen automatically, but lately there have been issues with
    # jobs restarting themselves, and maybe this line will help with that?
    wandb.finish()

    # Check if we have reached the end of the training
    complete = bool(pm.epoch == end_epochs[-1])
    return complete
