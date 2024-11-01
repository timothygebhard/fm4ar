"""
Script to draw samples from the prior and simulate spectra.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from fm4ar.datasets.vasist_2023.prior import THETA_0, Prior
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.utils.paths import expand_env_variables_in_path


def get_cli_arguments() -> argparse.Namespace:
    """
    Get command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-spectra",
        type=int,
        default=3,
        help="Number of spectra to generate.",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Parameters to sample from the prior. If None, all 16 parameters "
            "will be sampled (default). Parameters that are not sampled will "
            "be set to their 'default' values (from THETA_0)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=43,
        help="Random seed for sampling from prior.",
    )
    parser.add_argument(
        "--random-seed-offset",
        type=int,
        default=0,
        help="Offset for the random seed. Useful for parallelization.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1_000,
        choices=[400, 1_000],
        help=(
            "Spectral resolution (R = λ/∆λ). Default: 1000. Alternative: 400."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="$FM4AR_DATASETS_DIR/vasist-2023/output",
        help="Name of the output directory (default: 'output').",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=15,
        help="Simulation time limit per spectrum (in seconds).",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nSIMULATE SPECTRA\n", flush=True)

    args = get_cli_arguments()
    effective_random_seed = args.random_seed + args.random_seed_offset

    # Set up prior and simulator
    prior = Prior(random_seed=effective_random_seed)
    simulator = Simulator(R=args.resolution, time_limit=args.time_limit)

    # Define mask for parameters to fix to their THETA_0 values
    # In other words: The parameters given via `--parameters` are the ones
    # that will be sampled randomly from their respective prior.
    if args.parameters is None:
        mask = np.zeros(len(prior.names), dtype=bool)
    else:
        mask = np.array([p not in args.parameters for p in prior.names])

    # Print the "free" parameters (i.e., the ones that will be sampled)
    # We also need to encode the parameter names as variable-length strings
    # to be able to store them in the attributes of the output HDF file.
    params = np.array(prior.names)[~mask]
    encoded_params = np.array(params, dtype=h5py.special_dtype(vlen=bytes))
    print("Parameters that will be sampled:\n", params, "\n", flush=True)

    # Prepare lists to store the results
    wlen = np.empty(0)
    list_of_theta = []
    list_of_flux = []

    # Run the simulation
    print("Simulating spectra:", flush=True)
    for _ in tqdm(range(args.n_spectra), ncols=80, total=args.n_spectra):

        # Sample parameters from prior and set fixed parameters to THETA_0
        theta = prior.sample()
        if mask.any():
            theta[mask] = THETA_0[mask]

        # Convert to float32 (already *before* passing to simulator; otherwise
        # re-running the simulator on the saved theta values may yield slighly
        # different results for the fluxes)
        theta = theta.astype(np.float32)

        # Simulate spectrum; skip if simulation failed
        result = simulator(theta)
        if result is not None:
            wlen, flux = result
        else:
            continue

        # Only store the spectrum if it contains no NaNs or infinities
        if not np.isnan(flux).any() and not np.isinf(flux).any():
            list_of_theta.append(theta)
            list_of_flux.append(flux)

    # Convert lists to arrays
    theta = np.array(list_of_theta, dtype=np.float32)
    flux = np.array(list_of_flux, dtype=np.float32)

    print(f"\nNumber of successful simulations: {len(theta)}\n", flush=True)

    # Save everything to an HDF file
    print("Saving to an HDF file...", end=" ", flush=True)
    output_dir = expand_env_variables_in_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"random-seed_{effective_random_seed:07d}.hdf"
    with h5py.File(file_path, "w") as f:
        f.attrs["params"] = encoded_params
        f.create_dataset(name="theta", data=theta, dtype=np.float32)
        f.create_dataset(name="wlen", data=wlen, dtype=np.float32)
        f.create_dataset(name="flux", data=flux, dtype=np.float32)
    print("Done!\n", flush=True)

    print(f"This took {time.time() - script_start:.1f} seconds.\n")
