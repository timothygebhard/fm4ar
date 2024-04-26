"""
Collect quantile information for true parameter values.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.utils.hdf import save_to_hdf
from fm4ar.importance_sampling.utils import compute_effective_sample_size


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory."
    )
    parser.add_argument(
        "--name-pattern",
        type=str,
        required=True,
        help="Pattern for the name of the run directories to include."
    )

    args = parser.parse_args()
    return args


def get_quantile(
    value: float,
    array: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
) -> float:
    """
    Determine the quantile of the `value` in the weighted `array`.

    Args:
        value: The value for which to determine the quantile.
        array: The array of values.
        weights: The weights for the values in the array.
        bins: The bins to use for the quantile calculation.

    Returns:
        The quantile of the value in the weighted array.
    """

    # Compute a weighted histogram of the `array`
    hist, _ = np.histogram(array, bins=bins, weights=weights)

    # Determine the index of the bin containing the `value`
    idx = np.digitize(value, bins, right=True)

    # Compute the fraction of the histogram below the `value`
    # This is our estimate for the quantile in the weighted array
    quantile = float(np.sum(hist[:idx]) / np.sum(hist))

    return quantile


if __name__ == "__main__":

    script_start = time.time()
    print("\nCOLLECT QUANTILE INFORMATION\n", flush=True)

    # Get the command line arguments
    args = get_cli_arguments()

    # Determine the directories to include
    print("Collecting run directories...", end=" ", flush=True)
    important_sampling_dir = Path(args.experiment_dir) / "importance_sampling"
    run_dirs = sorted(
        filter(
            lambda path: (
                path.is_dir()
                and (path / "target_spectrum.hdf").exists()
                and (path / "importance_sampling_results.hdf").exists()
            ),
            important_sampling_dir.glob(args.name_pattern)
        )
    )
    print("Done!\n", flush=True)

    # Initialize the quantile information
    results: dict[str, list] = {
        # "run_dir": [],
        "sampling_efficiency": [],
        "quantiles_without_is": [],
        "quantiles_with_is": [],
    }

    # Loop over the run directories
    print("Collecting quantile information:", flush=True)
    for run_dir in tqdm(run_dirs, ncols=80, total=len(run_dirs)):

        # Read in the true parameter values
        with h5py.File(run_dir / "target_spectrum.hdf", "r") as f:
            theta_true = np.array(f["theta"])

        # Read in the importance sampling results
        with h5py.File(run_dir / "importance_sampling_results.hdf", "r") as f:
            samples = np.array(f["theta"])
            weights = np.array(f["weights"])

        # Compute the sampling efficiency
        _, sampling_efficiency = compute_effective_sample_size(weights)

        # Loop over the parameters to compute the quantiles
        quantiles_without_is = []
        quantiles_with_is = []
        for i in range(len(theta_true)):

            # Construct the bins
            # Going from the minimum to the maximum sample is probably fine,
            # because all other bins would be empty anyway and thus add 0
            bins = np.linspace(
                np.min(samples[:, i]),
                np.max(samples[:, i]),
                100,
            )

            # Compute the quantiles with and without importance sampling
            quantiles_without_is.append(
                get_quantile(
                    value=float(theta_true[i]),
                    array=samples[:, i],
                    weights=np.ones(len(samples)),
                    bins=bins,
                )
            )
            quantiles_with_is.append(
                get_quantile(
                    value=float(theta_true[i]),
                    array=samples[:, i],
                    weights=weights,
                    bins=bins,
                )
            )

        # Store the quantiles
        # results["run_dir"].append(run_dir)
        results["sampling_efficiency"].append(sampling_efficiency)
        results["quantiles_without_is"].append(quantiles_without_is)
        results["quantiles_with_is"].append(quantiles_with_is)

    print("\nSaving results to HDF...", end=" ", flush=True)

    # Convert the lists to arrays
    results_as_arrays = {}
    for key in results:
        results_as_arrays[key] = np.array(results[key])

    # Save the results to an HDF file
    file_path = important_sampling_dir / "quantile_information.hdf"
    save_to_hdf(file_path=file_path, **results_as_arrays)

    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
