"""
Create a corner plot with the original posterior and the importance
sampling posterior.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from fm4ar.datasets.vasist_2023.prior import LABELS
from fm4ar.nested_sampling.plotting import create_posterior_plot


if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE CORNER PLOT\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    args = parser.parse_args()

    # Load results from HDF
    print("Loading results...", end=" ", flush=True)
    file_path = args.experiment_dir / "importance_sampling_results.hdf"
    with h5py.File(file_path, "r") as hdf_file:
        theta = np.array(hdf_file["theta"])
        weights = np.array(hdf_file["weights"])
        theta_0 = np.array(hdf_file["theta_0"])
        parameter_mask = np.array(hdf_file["parameter_mask"])
    print("Done!")

    # Compute sample efficiency
    n_eff = np.sum(weights) ** 2 / np.sum(weights**2)
    sample_efficiency = n_eff / len(weights)

    # Create plot
    print("Creating plot...", end=" ", flush=True)
    create_posterior_plot(
        samples=theta,
        weights=weights,
        parameter_mask=parameter_mask,
        ground_truth=theta_0,
        names=np.array(LABELS)[parameter_mask].tolist(),
        sample_efficiency=sample_efficiency,
        experiment_dir=args.experiment_dir,
    )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds.\n")
