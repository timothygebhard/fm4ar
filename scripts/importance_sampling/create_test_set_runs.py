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

from fm4ar.importance_sampling.config import load_config
from fm4ar.utils.config import save_config
from fm4ar.utils.paths import expand_env_variables_in_path


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
            "Path to the base configuration file. The run directories "
            "will be created in the same directory as this file. This can "
            "be either an absolute path, or a path relative to the "
            "`importance_sampling` directory in the experiment directory."
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
        help="Path to the directory that holds the trained model."
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

    # Resolve the path to the base config in case it is not an absolute path
    if not args.base_config_path.is_absolute():
        args.base_config_path = expand_env_variables_in_path(
            Path(args.experiment_dir)
            / "importance_sampling"
            / args.base_config_path
        )

    # Get directory for the runs and load the base configuration
    runs_dir = args.base_config_path.parent
    file_name = args.base_config_path.name
    base_config = load_config(runs_dir, file_name)

    # Loop over indices
    for idx in range(args.start_idx, args.end_idx + 1):

        print(f"Creating config for index {idx}...", end=" ", flush=True)

        # Create a new configuration
        config = deepcopy(base_config)
        config.target_spectrum.file_path = expand_env_variables_in_path(
            args.target_file_path
        )
        config.target_spectrum.index = idx

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

        print("Done!", flush=True)

        # Launch the run
        if not args.no_launch:
            print(f"Launching run for index {idx}...", end=" ", flush=True)
            cmd = [
                sys.executable,
                launch_script.as_posix(),
                "--start-submission",
                "--experiment-dir",
                args.experiment_dir.as_posix(),
                "--working-dir",
                run_dir.as_posix()
            ]
            run(cmd, check=True, capture_output=True)
            print("Done!", flush=True)
            print()

    print(f"\nThis took {time.time() - script_start:.1f} sec!\n", flush=True)
