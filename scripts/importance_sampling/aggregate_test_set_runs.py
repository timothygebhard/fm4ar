"""
Aggregate information from test set runs (e.g., quantile information,
ranks, log evidence, ...) and save it to a single HDF file.
"""

import argparse
import time
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from p_tqdm import p_map
from scipy.spatial.distance import jensenshannon

from fm4ar.priors.base import BasePrior
from fm4ar.utils.distributions import compute_smoothed_histogram
from fm4ar.utils.hdf import load_from_hdf


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="vasist_2023",
        help="Name of the dataset (needed for prior boundaries)."
    )
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


def get_results_for_run_dir(
    run_dir: Path,
    prior: BasePrior,
) -> dict[str, np.ndarray]:
    """
    Aggregate the results for a single run directory. The `prior` is
    required to construct the bins for the JSD calculation.
    """

    # Read in the importance sampling results
    results = load_from_hdf(
        file_path=run_dir / "results.hdf",
        keys=[
            "log_evidence",
            "log_evidence_std",
            "log_prob_samples",
            "log_prob_theta_true",
            "samples",
            "sampling_efficiency",
            "simulation_efficiency",
            "weights",
        ]
    )
    results["run_dir"] = np.array(run_dir.as_posix(), dtype=object)

    # Read in the true parameter values
    with h5py.File(run_dir / "target_spectrum.hdf", "r") as f:
        results["sigma"] = np.array(f["sigma"])
        results["theta"] = np.array(f["theta"])

    # Compute the rank for the ground truth theta value
    results["rank_without_is"] = np.array(
        np.mean(results["log_prob_samples"] < results["log_prob_theta_true"])
    )
    results["rank_with_is"] = np.array(
        np.average(
            results["log_prob_samples"] < results["log_prob_theta_true"],
            weights=results["weights"],
        )
    )

    # Loop over the parameters to compute the quantiles
    results["quantiles_without_is"] = np.full(prior.ndim, np.nan)
    results["quantiles_with_is"] = np.full(prior.ndim, np.nan)
    for i in range(prior.ndim):

        # Construct the bins
        # Going from the minimum to the maximum sample is probably fine,
        # because all other bins would be empty anyway and thus add 0
        bins = np.linspace(
            np.min(results["samples"][:, i]),
            np.max(results["samples"][:, i]),
            101,
        )

        # Compute the quantiles with and without importance sampling
        results["quantiles_without_is"][i] = get_quantile(
            value=float(results["theta"][i]),
            array=results["samples"][:, i],
            weights=np.ones_like(results["weights"]),
            bins=bins,
        )
        results["quantiles_with_is"][i] = get_quantile(
            value=float(results["theta"][i]),
            array=results["samples"][:, i],
            weights=results["weights"],
            bins=bins,
        )

    # Compute the JSD (in mnat) between the marginal posterior distributions
    # with and without importance sampling
    results["jsd_with_without_is"] = np.full(prior.ndim, np.nan)
    for i in range(prior.ndim):

        # Construct the bins
        # We use the full prior range here because otherwise the results
        # might not be properly comparable between different runs
        bins = np.linspace(prior.lower[i], prior.upper[i], 101)

        _, hist_without_is = compute_smoothed_histogram(
            bins=bins,
            samples=results["samples"][:, i],
            weights=np.ones_like(results["weights"]),
            sigma=3,  # note: smoothing sigma!
        )
        _, hist_with_is = compute_smoothed_histogram(
            bins=bins,
            samples=results["samples"][:, i],
            weights=results["weights"],
            sigma=3,  # note: smoothing sigma!
        )
        jsd = 1000 * jensenshannon(hist_without_is, hist_with_is)
        results["jsd_with_without_is"][i] = jsd

    # Drop keys that we do not need for the aggregated results
    del results["samples"]
    del results["weights"]

    return results


if __name__ == "__main__":

    script_start = time.time()
    print("\nAGGREGATE RESULTS FROM TEST SET RUNS\n", flush=True)

    # Get the command line arguments
    args = get_cli_arguments()

    # Load the prior for the dataset
    if args.dataset == "vasist_2023":
        from fm4ar.datasets.vasist_2023.prior import Prior
    else:
        raise ValueError("Unknown dataset!")

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
    print("Aggregating results:", flush=True)
    list_of_dicts = p_map(
        partial(
            get_results_for_run_dir,
            prior=Prior(random_seed=0),
        ),
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
                    dtype=np.float32,
                )
            else:
                dataset = f.create_dataset(
                    name=key,
                    shape=(len(value),),
                    dtype=h5py.special_dtype(vlen=str),
                )
                dataset[:] = list(map(str, value))

    print("Done!\n", flush=True)
    print("Results saved to:\n", file_path, flush=True)

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
