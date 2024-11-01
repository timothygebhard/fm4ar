"""
Replace the config in a checkpoint with a new config.
"""

import argparse
import time
from copy import deepcopy
from pathlib import Path
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
        help="Name of checkpoint file for which to update the config.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup of the old checkpoint.",
    )
    parser.add_argument(
        "--no-confirmation",
        action="store_true",
        help="Skip confirmation and update the config in the checkpoint.",
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

    # Add the theta_dim and context_dim to the model settings
    # These are added in the `prepare_new()` method when starting a new
    # training run, and dropping them here would cause an error when building
    # the model from the checkpoint file in `prepare_resume()`.
    new_config["model"]["dim_theta"] = old_config["model"]["dim_theta"]
    new_config["model"]["dim_context"] = old_config["model"]["dim_context"]

    # Compute the difference between the old and the new config
    print("Difference between old and new config:\n")
    diff = DeepDiff(old_config, new_config)
    for line in diff.pretty().split("\n"):
        print(f"  {line}")
    print("\n")

    # Ask for confirmation
    if not args.no_confirmation:
        if not confirm("Do you want to update the config in the checkpoint?"):
            print("\nAborting!\n")
            exit(0)
        print()
    else:
        print("Updating the config in the checkpoint without confirmation!\n")

    # Create a backup of the old checkpoint
    if not args.no_backup:
        print("Creating backup of old checkpoint...", end=" ", flush=True)
        backup_path = file_path.with_suffix(".backup.pt")
        copyfile(file_path, backup_path)
        print("Done!", flush=True)
    else:
        print("Skipping backup of old checkpoint!", flush=True)

    # Update config and save checkpoint
    print("Updating config in checkpoint...", end=" ", flush=True)
    checkpoint["config"] = new_config
    torch.save(checkpoint, file_path)
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start_time:.2f} seconds!\n")
