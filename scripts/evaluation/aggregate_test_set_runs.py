"""
Aggregate information from test set runs (e.g., quantile information,
but also ranks, log evidence, ...) and save it to a single HDF file.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.importance_sampling.utils import (
    compute_effective_sample_size,
    compute_log_evidence,
)


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
    print("\nAGGREGATE RESULTS FROM TEST SET RUNS\n", flush=True)

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

    # Initialize variables that will hold the results that we collect
    results: dict[str, list] = {
        "run_dir": [],
        "log_evidence": [],
        "log_evidence_std": [],
        "sampling_efficiency": [],
        "quantiles_without_is": [],
        "quantiles_with_is": [],
        "theta": [],
        "sigma": [],
        "snr": [],
        "flux": [],
        "rank_with_is": [],
        "rank_without_is": [],
    }

    # Loop over the run directories
    print("Collecting quantile information:", flush=True)
    for run_dir in tqdm(run_dirs, ncols=80, total=len(run_dirs)):

        # Read in the true parameter values
        with h5py.File(run_dir / "target_spectrum.hdf", "r") as f:
            sigma = float(np.array(f["sigma"]))
            snr = float(np.array(f["snr"]))
            theta = np.array(f["theta"])
            flux = np.array(f["flux"])

        # Read in the importance sampling results
        with h5py.File(run_dir / "importance_sampling_results.hdf", "r") as f:
            if "theta" in f.keys():  # For the old results
                samples = np.array(f["theta"])
            else:
                samples = np.array(f["samples"])
            weights = np.array(f["weights"])
            raw_log_weights = np.array(f["raw_log_weights"])
            log_prob_samples = np.array(f["log_prob_samples"])
            log_prob_theta_true = np.array(f["log_prob_theta_true"])

        # Compute the rank for the ground truth theta value
        rank_without_is = np.mean(log_prob_samples < log_prob_theta_true)
        rank_with_is = np.average(
            log_prob_samples < log_prob_theta_true,
            weights=weights,
        )
        results["rank_without_is"].append(rank_without_is)
        results["rank_with_is"].append(rank_with_is)

        # Compute the sampling efficiency and the evidence
        _, sampling_efficiency = compute_effective_sample_size(weights)
        log_evidence, log_evidence_std = compute_log_evidence(raw_log_weights)

        # Loop over the parameters to compute the quantiles
        quantiles_without_is = []
        quantiles_with_is = []
        for i in range(len(theta)):

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
                    value=float(theta[i]),
                    array=samples[:, i],
                    weights=np.ones(len(samples)),
                    bins=bins,
                )
            )
            quantiles_with_is.append(
                get_quantile(
                    value=float(theta[i]),
                    array=samples[:, i],
                    weights=weights,
                    bins=bins,
                )
            )

        # Store the quantiles
        results["run_dir"].append(run_dir.as_posix())
        results["log_evidence"].append(log_evidence)
        results["log_evidence_std"].append(log_evidence_std)
        results["theta"].append(theta)
        results["sigma"].append(sigma)
        results["snr"].append(snr)
        results["flux"].append(flux)
        results["sampling_efficiency"].append(sampling_efficiency)
        results["quantiles_without_is"].append(quantiles_without_is)
        results["quantiles_with_is"].append(quantiles_with_is)

    print("\nSaving results to HDF...", end=" ", flush=True)

    # Save the results to an HDF file
    file_name = "aggregated__" + args.name_pattern.replace("*", "X") + ".hdf"
    file_path = important_sampling_dir / file_name
    with h5py.File(file_path, "w") as f:
        for key, value in results.items():
            if key != "run_dir":
                f.create_dataset(
                    name=key,
                    data=np.array(value),
                    dtype=np.float32
                )
            else:
                dataset = f.create_dataset(
                    name=key,
                    shape=(len(value),),
                    dtype=h5py.special_dtype(vlen=str),
                )
                dataset[:] = value

    print("Done!", flush=True)
    print("Results saved to:", file_path, flush=True)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
