"""
Create experiments for a hyperparameter sweep and launch the jobs.
"""

import argparse
import sys
import time
from functools import reduce
from itertools import product
from pathlib import Path
from subprocess import run
from typing import Any

import click
import numpy as np

from fm4ar.utils.config import load_config, save_config


def get_arguments() -> argparse.Namespace:
    """
    Collect command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Base directory where the sweep experiments will be created.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Path to the directory with the config files for the sweep.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="If set, list but do not create or launch sweep experiments.",
    )
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=10,
        help=(
            "Number of experiments to create. If None, all combinations of "
            "hyperparameters will be created. If a number is given, the "
            "combinations will be chosen randomly."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the random number generator.",
    )
    args = parser.parse_args()

    return args


def assert_key_exists(d: dict, key: str) -> None:
    """
    Ensure a key given as "a/b/c" exists in a nested dictionary.
    """

    try:
        reduce(lambda d, k: d[k], key.split("/"), d)
    except KeyError as e:
        raise KeyError(f"Key '{key}' not found in base config!") from e


def set_value_in_nested_dict(d: dict, key: str, value: Any) -> None:
    """
    Set a value in a nested dictionary given as "a/b/c".
    """

    keys = key.split("/")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


if __name__ == "__main__":
    script_start = time.time()
    print("\nCREATE HYPERPARAMETER SWEEP\n")

    # Get command line arguments
    args = get_arguments()

    # Create a new random number generator
    rng = np.random.RandomState(args.random_seed)

    # Get the config files for the sweep
    base_config = load_config(args.config_dir, "base.yaml")
    sweep_config = load_config(args.config_dir, "sweep.yaml")

    # Ensure that the sweep structure is valid
    for kwargs in sweep_config.values():
        assert_key_exists(base_config, kwargs["key"])

    # Ensure that all experiments will be placed in a group in wandb
    base_config["local"]["wandb"]["group"] = args.config_dir.name

    # Create all combinations of hyperparameters
    combinations = list(product(*[v["values"] for v in sweep_config.values()]))
    print(f"Total number of all combinations: {len(combinations)}\n")

    # If a number of experiments is given, choose a random subset
    if args.n_experiments is not None:
        idx = rng.choice(len(combinations), args.n_experiments, replace=False)
        subset = sorted([combinations[i] for i in idx])
    else:
        subset = sorted(combinations)

    # Print the set of combinations for which we will create experiments
    print(f"The following {args.n_experiments} combinations were chosen:\n")
    for combination in subset:
        joined = "\t".join(
            [
                f"{k}={v}"
                for k, v in zip(sweep_config.keys(), combination, strict=True)
            ]
        )
        print("  " + joined)

    # In case of a dry run, we are done here
    if args.dry:
        print("\nThis was a dry run, no experiments were created.\n")
        sys.exit(0)

    # Double-check if we really want to continue
    if not click.confirm("\nDo you want to continue?", default=False):
        print("\nLaunch aborted!\n")
        sys.exit(0)

    # Create the target directory if it does not exist
    # The target directory is placed in the `base_dir` and matches the name
    # of the config directory, e.g., `fm-sweep-1`.
    target_dir = args.base_dir / args.config_dir.name

    # Create the experiments
    print("\nCreating experiment directories:\n")
    experiment_dirs = []
    for combination in subset:

        # Define experiment name
        parts = []
        for p, v in zip(sweep_config.values(), combination, strict=True):
            parts.append(f"{v:.1e}" if p["name"] == "lr" else str(v))
        experiment_name = "__".join(parts)

        # Create the experiment directory
        experiment_dir = args.base_dir / args.config_dir.name / experiment_name
        experiment_dir.mkdir(parents=True)
        experiment_dirs.append(experiment_dir)
        print("  " + experiment_dir.as_posix())

        # Create the config file by updating the base config
        config = base_config.copy()
        for parameter, value in zip(sweep_config, combination, strict=True):
            set_value_in_nested_dict(
                d=config,
                key=sweep_config[parameter]["key"],
                value=value,
            )

        # Ensure the folder name matches the name on wandb
        config["local"]["wandb"]["name"] = experiment_dir.name

        save_config(config, experiment_dir, "config.yaml")

    # Get path to script for launching experiments on the cluster
    script_path = Path(__file__).parents[1] / "training" / "train_cluster.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Launch script '{script_path}' not found!")

    # Launch the experiments
    print("\nLaunching experiments:\n")
    for experiment_dir in experiment_dirs:
        cmd = [
            sys.executable,
            script_path.as_posix(),
            "--experiment-dir",
            experiment_dir.as_posix(),
            "--start-submission",
        ]
        process = run(cmd, capture_output=True)
        print(process.stdout.decode("utf-8"))

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")
