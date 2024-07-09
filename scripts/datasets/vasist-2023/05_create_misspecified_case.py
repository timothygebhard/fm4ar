"""
Create conservative and misspecified retrieval cases used to test if
the sampling efficiency can be used to flag misspecified noise levels.
"""

import argparse
import datetime
import sys
import time
from importlib.metadata import version

import h5py
import numpy as np

from fm4ar.datasets.vasist_2023.prior import THETA_0
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.utils.environment import get_packages
from fm4ar.utils.git_utils import get_git_hash
from fm4ar.utils.paths import get_datasets_dir

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE MISSPECIFIED RETRIEVAL CASES\n", flush=True)

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        default=400,
        choices=[400, 1_000],
        help="Spectral resolution (R = λ/∆λ). Default: 400.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=15,
        help="Time limit for the simulation in seconds. Default: 15.",
    )
    args = parser.parse_args()

    # Set up simulator
    print("Set up simulator...", end=" ", flush=True)
    simulator = Simulator(R=args.resolution, time_limit=args.time_limit)
    print("Done!", flush=True)

    # Simulate noise-free benchmark spectrum
    print("Simulating benchmark spectrum...", end=" ", flush=True)
    result = simulator(THETA_0.astype(np.float32))
    if result is None:
        print("Failed!", flush=True)
        raise RuntimeError("Simulation failed!")
    else:
        wlen, flux = result
        wlen = wlen.astype(np.float32)
        flux = flux.astype(np.float32)
        print("Done!\n", flush=True)

    # Prepare output directory
    output_dir = get_datasets_dir() / "vasist-2023" / "misspecified"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather meta-information to save to HDF file (as attributes)
    n_bins = len(wlen)
    prt_version = version("petitRADTRANS")
    metadata = {
        "HEAD of fm4ar": get_git_hash(),
        "Timestamp (UTC)": datetime.datetime.utcnow().isoformat(),
        "Python version": sys.version,
        "petitRADTRANS version": version("petitRADTRANS"),
        "packages": "\n".join(get_packages()),
    }

    # Set up random number generator
    rng = np.random.default_rng(seed=42)

    # We create two versions: one with actual noise level 0.1 and assumed
    # noise level 0.5, and one with actual noise level 0.5 and assumed noise
    # level 0.1. This is to demonstrate the effect of misspecifying the noise
    # level in the retrieval.
    for prefix, sigma_actual, sigma_assumed in [
        ("conservative", 0.1, 0.5),
        ("misspecified", 0.5, 0.1),
    ]:

        print(f"Creating {prefix} retrieval case...", end=" ", flush=True)

        # Define *assumed* error bars for the benchmark spectrum
        error_bars = np.full_like(flux, sigma_assumed, dtype=np.float32)

        # Sample random noise according to the *actual* noise level
        noise = rng.normal(scale=sigma_actual, size=n_bins).astype(np.float32)
        noisy_flux = flux + noise

        # Save the result
        file_name = prefix + f"__R-{args.resolution}__pRT-{prt_version}.hdf"
        file_path = output_dir / file_name
        with h5py.File(file_path, "w") as f:
            f.attrs.update(metadata)
            f.create_dataset("wlen", data=wlen.reshape(1, -1))
            f.create_dataset("flux", data=noisy_flux.reshape(1, -1))
            f.create_dataset("noise", data=noise)
            f.create_dataset("theta", data=THETA_0.reshape(1, -1))
            f.create_dataset("error_bars", data=error_bars.reshape(1, -1))

        print("Done!", flush=True)

    print("\nResults saved to:", output_dir, flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
