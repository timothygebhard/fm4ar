"""
Create importance sampling configurations for the test set, and launch
the corresponding runs.
"""

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path
from subprocess import run

import h5py
import numpy as np

from fm4ar.importance_sampling.config import load_config
from fm4ar.utils.config import save_config
from fm4ar.utils.hdf import save_to_hdf


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-config-path",
        type=Path,
        required=True,
        help=(
            "Path to the base configuration file. The run directories will "
            "be created in the same directory as this file."
        )
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=10,
        help="Last index in the target file."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory."
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="If set, create but do not launch the runs."
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="First index in the target file."
    )
    parser.add_argument(
        "--target-file-path",
        type=Path,
        required=True,
        help="Path to the HDF file containing the target spectra."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE TEST SET IMPORTANCE SAMPLING RUNS\n", flush=True)

    # Get the command line arguments
    args = get_cli_arguments()

    # Get path to launch script
    launch_script = Path(__file__).parent / "run_importance_sampling.py"

    # Load the base configuration and get directory for the runs
    base_config = load_config(args.base_config_path)
    runs_dir = args.base_config_path.parent

    # Loop over indices
    for idx in range(args.start_idx, args.end_idx + 1):

        print(f"Creating config for index {idx}...", end=" ", flush=True)

        # Load the target spectrum and all associated information
        target_spectrum = {}
        with h5py.File(args.target_file_path, "r") as f:
            target_spectrum["wlen"] = np.array(f["wlen"][0])
            target_spectrum["flux"] = np.array(f["flux"][idx])
            target_spectrum["noise"] = np.array(f["noise"][idx])
            target_spectrum["theta"] = np.array(f["theta"][idx])
            target_spectrum["sigma"] = np.array(f["sigma"][idx])

        # Create a new configuration
        config = deepcopy(base_config)
        config.target_spectrum.file_path = args.target_file_path.as_posix()
        config.target_spectrum.index = idx
        config.likelihood.sigma = float(target_spectrum["sigma"])

        # Create the directory for the run
        target_stem = args.target_file_path.stem
        run_dir = runs_dir / f"{target_stem}__idx-{idx:04d}"
        run_dir.mkdir(exist_ok=True)

        # Save the configuration
        save_config(
            config=config.dict(),
            experiment_dir=run_dir,
            file_name="importance_sampling.yaml",
        )

        # Save a backup of the target spectrum
        save_to_hdf(
            file_path=run_dir / "target_spectrum.hdf",
            **target_spectrum,
        )

        print("Done!", flush=True)

        # Launch the run
        if not args.no_launch:
            print(f"Launching run for index {idx}...", end=" ", flush=True)
            cmd = [
                sys.executable,
                launch_script.as_posix(),
                "--start-submission",
                "--working-dir",
                run_dir.as_posix()
            ]
            run(cmd, check=True, capture_output=True)
            print("Done!", flush=True)
            print()

    print(f"\nThis took {time.time() - script_start:.1f} sec!\n", flush=True)
