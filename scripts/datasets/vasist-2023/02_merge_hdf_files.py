"""
Merge the HDF files for each random seed into a single HDF file.
"""

import argparse
import time
from pathlib import Path

import h5py

from fm4ar.utils.paths import expand_env_variables_in_path
from fm4ar.utils.hdf import merge_hdf_files


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete-after-merge",
        default=False,
        action="store_true",
        help="Delete the source HDF files after merging? Default: False.",
    )
    parser.add_argument(
        "--name-pattern",
        type=str,
        default="random-seed_*.hdf",
        help="Name pattern for the HDF files (default: random-seed_*.hdf).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="merged.hdf",
        help="Name of the output HDF file (default: merged.hdf).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default="$FM4AR_DATASETS_DIR/vasist-2023/output",
        help="Path to the target directory contain HDF files to be merged.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nMERGE HDF FILES\n", flush=True)

    args = get_cli_arguments()

    # Construct target dir and output file path
    target_dir = expand_env_variables_in_path(args.target_dir)
    output_file_path = target_dir / args.output_name

    # Merge the HDF files
    print("Merging HDF files:", flush=True)
    merge_hdf_files(
        target_dir=target_dir,
        name_pattern=args.name_pattern,
        output_file_path=output_file_path,
        singleton_keys=("wlen", ),
        delete_after_merge=args.delete_after_merge,
        show_progressbar=True,
    )

    # Print total number of spectra after merging
    with h5py.File(output_file_path, "r") as f:
        n_spectra = f["flux"].shape[0]
    print(f"\nTotal number of spectra after merging: {n_spectra:,}")

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
