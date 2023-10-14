"""
Utility script to rename datasets in an HDF file.
"""

import argparse
from pathlib import Path

import h5py


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        type=Path,
        required=True,
        help="Path to the HDF file.",
    )
    parser.add_argument(
        "--mappings",
        type=str,
        nargs="+",
        required=True,
        help="List of mappings in the form 'old_name=new_name'.",
    )
    args = parser.parse_args()

    return args


def rename_dataset(
    file_path: Path,
    old_name: str,
    new_name: str,
) -> None:
    """
    Rename a dataset in an HDF file.
    """

    with h5py.File(file_path, "r+") as f:

        # If the dataset does not exist, just skip it
        if old_name not in f:
            print(f"Dataset '{old_name}' not found in '{file_path}'!")
            return

        # Otherwise, "rename" the dataset
        print(f"Renaming '{old_name}' to '{new_name}'...", end=" ", flush=True)
        f.copy(old_name, new_name)
        del f[old_name]
        print("Done!")


if __name__ == "__main__":

    args = get_cli_arguments()

    # Rename datasets
    for mapping in args.mappings:
        old_name, new_name = mapping.split("=")
        rename_dataset(
            file_path=args.file_path,
            old_name=old_name,
            new_name=new_name,
        )
