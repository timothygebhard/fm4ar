"""
Replace the config in a checkpoint with a new config.
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from shutil import copyfile

import torch
from click import confirm
from deepdiff.diff import DeepDiff

from fm4ar.utils.config import load_config


if __name__ == "__main__":

    script_start_time = time.time()
    print("\nUPDATE CONFIG IN CHECKPOINT\n")

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="model__latest.pt",
        help="Name of checkpoint file for which to update the config."
    )
    args = parser.parse_args()

    # Load the current config from the experiment directory
    print("Loading config from experiment directory...", end=" ")
    new_config = load_config(args.experiment_dir)
    print("Done!")

    # Load the checkpoint
    print("Loading checkpoint...", end=" ")
    file_path = args.experiment_dir / args.checkpoint_file
    checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
    old_config = deepcopy(checkpoint["config"])
    print("Done!\n")

    # Compute the difference between the old and the new config
    print("Difference between old and new config:")
    diff = DeepDiff(old_config, new_config)
    pprint(diff, indent=2)
    print()

    # Ask for confirmation
    if not confirm("Do you want to update the config in the checkpoint?"):
        print("Aborting!")
        exit(0)
    print()

    # Create a backup of the old checkpoint
    print("Creating backup of old checkpoint...", end=" ")
    backup_path = file_path.with_suffix(".backup.pt")
    copyfile(file_path, backup_path)
    print("Done!")

    # Update config and save checkpoint
    print("Updating config in checkpoint...", end=" ")
    checkpoint["config"] = new_config
    torch.save(checkpoint, file_path)
    print("Done!")

    print(f"\nThis took {time.time() - script_start_time:.2f} seconds!\n")
