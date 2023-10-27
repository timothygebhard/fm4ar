"""
Methods to prepare a new or resumed training run.
"""

from pathlib import Path

import wandb

from fm4ar.datasets import load_dataset, ArDataset
from fm4ar.models.base import Base
from fm4ar.models.build_model import build_model
from fm4ar.training.wandb import get_wandb_id
from fm4ar.utils.torchutils import get_number_of_model_parameters


def prepare_new(
    experiment_dir: Path,
    config: dict,
) -> tuple[Base, ArDataset]:
    """
    Prepare a new training run, that is, load the dataset and initialize
    a new posterior model according to the settings in the `config.yaml`
    file in the `experiment_dir`.

    Args:
        experiment_dir: Path to the experiment directory.
        config: Full experiment configuration.

    Returns:
        A tuple, `(pm, dataset)`, where `pm` is the posterior model
        and `dataset` is the dataset.
    """

    # Load the dataset
    name = config["data"]["name"]
    print(f"Loading dataset '{name}'...", end=" ")
    dataset = load_dataset(config=config)
    print("Done!", flush=True)

    # Add the theta_dim and context_dim to the model settings
    config["model"]["theta_dim"] = dataset.theta_dim
    config["model"]["context_dim"] = dataset.context_dim

    # Initialize the posterior model
    print("Building model from configuration...", end=" ")
    pm = build_model(
        experiment_dir=experiment_dir,
        config=config,
        device=config["local"]["device"],
    )
    print(f"Done! (device: {pm.device})")

    # Initialize Weights & Biases (if desired)
    if pm.use_wandb:
        print("\n\nInitializing Weights & Biases:")

        # Add number of model parameters to the config
        augmented_config = config.copy()
        augmented_config["n_model_parameters"] = {
            "trainable": get_number_of_model_parameters(pm.model, (True,)),
            "fixed": get_number_of_model_parameters(pm.model, (False,)),
            "total": get_number_of_model_parameters(pm.model),
        }

        # Add the experiment directory to the config
        augmented_config["experiment_dir"] = experiment_dir.as_posix()

        # Initialize Weights & Biases; this will produce some output to stderr
        wandb_id = get_wandb_id(experiment_dir)
        wandb.init(
            id=wandb_id,
            config=augmented_config,
            dir=experiment_dir,
            **config["local"]["wandb"],
        )

        # Define metrics
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        # Save the name of the run to a file in the experiment directory
        if wandb.run is not None:
            (experiment_dir / wandb.run.name).touch(exist_ok=True)

        print()

    return pm, dataset


def prepare_resume(
    experiment_dir: Path,
    checkpoint_name: str,
    config: dict,
    update_config: bool = False,
) -> tuple[Base, ArDataset]:
    """
    Prepare a training run by resuming from a checkpoint, that is, load
    the dataset, and instantiate the posterior model, optimizer and
    scheduler from the checkpoint.

    Args:
        experiment_dir: Path to the experiment directory.
        checkpoint_name: Name of the checkpoint file.
        config: Full experiment configuration.
        update_config: If `True`, replace the configuration in the model
            checkpoint with the configuration from the experiment with
            the `config` passed to this function. This can be useful for
            small bug fixes; however, if the updated configuration is
            not compatible with the model in the checkpoint, this will
            lead to an error.

    Returns:
        A tuple, `(pm, dataset)`, where `pm` is the posterior model
        and `dataset` is the dataset.
    """

    # Instantiate the posterior model
    print("Building model from checkpoint...", end=" ")
    file_path = experiment_dir / checkpoint_name
    pm = build_model(
        experiment_dir=experiment_dir,
        file_path=file_path,
        device=config["local"]["device"],
    )
    print("Done!", flush=True)

    # Load the dataset (using config from checkpoint)
    name = pm.config["data"]["name"]
    print(f"Loading dataset '{name}'...", end=" ")
    dataset = load_dataset(config=pm.config)
    print("Done!", flush=True)

    # Reload the configuration from the experiment directory if desired
    if update_config:
        pm.config = config
        print("Updated model configuration!")

    # Initialize Weights & Biases; this will produce some output to stderr
    if config["local"].get("wandb", False):
        print("\n\nRe-initializing Weights & Biases:", flush=True)

        wandb_id = get_wandb_id(experiment_dir)
        wandb.init(
            id=wandb_id,
            resume="must",
            dir=experiment_dir,
            **config["local"]["wandb"],
        )

    return pm, dataset
