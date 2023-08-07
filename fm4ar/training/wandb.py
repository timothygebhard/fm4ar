"""
Utility functions for Weights & Biases (wandb).
"""

from pathlib import Path

from wandb.util import generate_id


def get_wandb_id(experiment_dir: Path) -> str:
    """
    Check if the experiment directory contains a Weights & Biases ID.
    If yes, return it, otherwise create a new one.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        The wandb ID for the experiment.
    """

    # Create the wandb directory inside the experiment directory
    wandb_dir = experiment_dir / "wandb"
    wandb_dir.mkdir(exist_ok=True)

    # Check if the wandb directory already contains a wandb ID...
    file_path = wandb_dir / "wandb_id"
    if file_path.exists():
        with open(file_path) as file:
            wandb_id = file.read().strip()

    # ...otherwise, create a new one
    else:
        wandb_id = generate_id()
        with open(file_path, "w") as file:
            file.write(wandb_id)

    return wandb_id
