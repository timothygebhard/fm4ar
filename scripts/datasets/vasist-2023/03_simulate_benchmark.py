"""
Simulate default benchmark spectrum for the `vasist_2023` dataset.
"""

import argparse
import datetime
import sys
import time
from importlib.metadata import version

import h5py
import numpy as np

from fm4ar.datasets.vasist_2023.prior import SIGMA, THETA_0
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.utils.environment import get_packages
from fm4ar.utils.git_utils import get_git_hash
from fm4ar.utils.paths import get_datasets_dir

if __name__ == "__main__":

    script_start = time.time()
    print("\nSIMULATE VASIST-2023 BENCHMARK SPECTRUM\n", flush=True)

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

    # Simulate benchmark spectrum
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
    output_dir = get_datasets_dir() / "vasist-2023" / "benchmark"
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

    # Create versions for different noise levels
    # We use the default noise level from  Vasist et al. (2023), i.e.,
    # σ = 0.125754 x 10^-16 W / m^2 / µm, as as well as other levels
    for sigma in [0.1, SIGMA, 0.2, 0.3, 0.4]:

        print(f"Creating files for sigma = {sigma}:", flush=True)

        # Define error bars for the benchmark spectrum
        error_bars = np.full_like(flux, sigma, dtype=np.float32)

        # First, save the noise-free benchmark spectrum
        print("--Saving noise-free benchmark spectrum...", end=" ", flush=True)
        file_name = (
            f"noise-free__"
            f"sigma-{sigma}__"
            f"R-{args.resolution}__"
            f"pRT-{prt_version}.hdf"
        )
        file_path = output_dir / file_name
        with h5py.File(file_path, "w") as f:
            f.attrs.update(metadata)
            f.create_dataset("wlen", data=wlen.reshape(1, -1))
            f.create_dataset("flux", data=flux.reshape(1, -1))
            f.create_dataset("theta", data=THETA_0.reshape(1, -1))
            f.create_dataset("error_bars", data=error_bars.reshape(1, -1))
        print("Done!", flush=True)

        # Next, create noisy versions of the benchmark spectrum
        # We create 100 noise realizations using the
        print("--Creating noisy versions...", end=" ", flush=True)
        noise = rng.normal(scale=sigma, size=(100, n_bins)).astype(np.float32)
        noisy_flux = flux + noise
        print("Done!", flush=True)

        # Save noisy versions of the benchmark spectrum
        print("--Saving noisy versions...", end=" ", flush=True)
        file_name = (
            f"with-noise__"
            f"sigma-{sigma}__"
            f"R-{args.resolution}__"
            f"pRT-{prt_version}.hdf"
        )
        file_path = output_dir / file_name
        with h5py.File(file_path, "w") as f:
            f.attrs.update(metadata | {"noise_level": SIGMA})
            f.create_dataset("wlen", data=wlen.reshape(1, -1))
            f.create_dataset("flux", data=noisy_flux)
            f.create_dataset("error_bars", data=np.tile(error_bars, (100, 1)))
            f.create_dataset("noise", data=noise)
            f.create_dataset("theta", data=np.tile(THETA_0, (100, 1)))
        print("Done!\n", flush=True)

    print("Results saved to:", output_dir, flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
