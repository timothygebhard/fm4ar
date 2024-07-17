"""
Aggregate information from test set runs (e.g., quantile information,
but also ranks, log evidence, ...) and save it to a single HDF file.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from p_tqdm import p_map
from scipy.spatial.distance import jensenshannon

from fm4ar.datasets.vasist_2023.prior import Prior
from fm4ar.utils.distributions import compute_smoothed_histogram


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name-pattern",
        type=str,
        required=True,
        help="Pattern for the name of the run directories to include."
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=16,
        help="Number of processes to use for parallelization."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Path to the directory that holds the test set runs."
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


def get_results_for_run_dir(run_dir: Path) -> dict:
    """
    Aggregate the results for a single run directory.
    """

    # Read in the true parameter values
    with h5py.File(run_dir / "target_spectrum.hdf", "r") as f:
        sigma = float(np.array(f["sigma"]))
        theta = np.array(f["theta"])
        flux = np.array(f["flux"])

    # Read in the importance sampling results
    with h5py.File(run_dir / "results.hdf", "r") as f:
        samples = np.array(f["samples"])
        weights = np.array(f["weights"])
        log_prob_samples = np.array(f["log_prob_samples"])
        log_prob_theta_true = np.array(f["log_prob_theta_true"])
        sampling_efficiency = np.array(f["sampling_efficiency"])
        simulation_efficiency = np.array(f["simulation_efficiency"])
        log_evidence = np.array(f["log_evidence"])
        log_evidence_std = np.array(f["log_evidence_std"])

    # Compute the rank for the ground truth theta value
    rank_without_is = np.mean(log_prob_samples < log_prob_theta_true)
    rank_with_is = np.average(
        log_prob_samples < log_prob_theta_true,
        weights=weights,
    )

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
            101,
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

    # Compute the JSD (in mnat) between the marginal posterior distributions
    # with and
    # without importance sampling
    jsd_with_without_is = []
    prior = Prior()
    for i in range(len(theta)):

        bins = np.linspace(prior.lower[i], prior.upper[i], 101)
        _, hist_without_is = compute_smoothed_histogram(
            bins=bins,
            samples=samples[:, i],
            weights=np.ones(len(samples)),
            sigma=3,  # note: smoothing sigma!
        )
        _, hist_with_is = compute_smoothed_histogram(
            bins=bins,
            samples=samples[:, i],
            weights=weights,
            sigma=3,  # note: smoothing sigma!
        )
        jsd = 1000 * jensenshannon(hist_without_is, hist_with_is)
        jsd_with_without_is.append(jsd)

    return {
        "flux": flux,
        "jsd_with_without_is": jsd_with_without_is,
        "log_evidence": log_evidence,
        "log_evidence_std": log_evidence_std,
        "quantiles_with_is": quantiles_with_is,
        "quantiles_without_is": quantiles_without_is,
        "rank_with_is": rank_with_is,
        "rank_without_is": rank_without_is,
        "run_dir": run_dir.as_posix(),
        "sampling_efficiency": sampling_efficiency,
        "sigma": sigma,
        "simulation_efficiency": simulation_efficiency,
        "theta": theta,
    }


if __name__ == "__main__":

    script_start = time.time()
    print("\nAGGREGATE RESULTS FROM TEST SET RUNS\n", flush=True)

    # Get the command line arguments
    args = get_cli_arguments()

    # Determine the directories to include
    print("Collecting run directories...", end=" ", flush=True)
    run_dirs = sorted(
        filter(
            lambda path: (
                path.is_dir()
                and (path / "target_spectrum.hdf").exists()
                and (path / "results.hdf").exists()
            ),
            args.runs_dir.glob(args.name_pattern)
        )
    )
    print("Done!\n", flush=True)

    # Loop over the run directories and collect the results in parallel
    print("Collecting quantile information:", flush=True)
    list_of_dicts = p_map(
        get_results_for_run_dir,
        run_dirs,
        num_cpus=args.n_processes,
        ncols=80,
    )

    # Convert list of dicts to dict of lists (for easier HDF saving)
    dict_of_lists = {
        key: [item[key] for item in list_of_dicts]
        for key in list_of_dicts[0]
    }

    # Save the results to an HDF file
    print("\nSaving results to HDF...", end=" ", flush=True)
    file_name = "aggregated__" + args.name_pattern.replace("*", "X") + ".hdf"
    file_path = args.runs_dir / file_name
    with h5py.File(file_path, "w") as f:
        for key, value in dict_of_lists.items():
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

    print("Done!\n", flush=True)
    print("Results saved to:\n", file_path, flush=True)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
