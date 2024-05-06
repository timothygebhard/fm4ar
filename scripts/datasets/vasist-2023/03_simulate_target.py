"""
Simulate default target spectrum for the `vasist_2023` dataset.
"""

import argparse
import time
from importlib.metadata import version

from fm4ar.utils.hdf import save_to_hdf
from fm4ar.datasets.vasist_2023.prior import THETA_0
from fm4ar.datasets.vasist_2023.simulator import Simulator
from fm4ar.utils.paths import get_datasets_dir


if __name__ == "__main__":

    script_start = time.time()
    print("\nSIMULATE TARGET DATA\n", flush=True)

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        default=1_000,
        choices=[400, 1_000],
        help="Spectral resolution (R = λ/∆λ). Default: 1000.",
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
    print("Done!")

    # Simulate target data
    print("Simulating target spectrum...", end=" ", flush=True)
    result = simulator(THETA_0)
    if result is None:
        print("Failed!")
        raise RuntimeError("Simulation failed!")
    else:
        wlen, flux = result
        print("Done!")

    # Save target data to HDF file
    # We add the version of petitRADTRANS to the file name in case we ever
    # need to know which version was used to generate the target spectrum
    print("Saving target spectrum...", end=" ", flush=True)
    output_dir = get_datasets_dir() / "vasist-2023" / "target"
    output_dir.mkdir(parents=True, exist_ok=True)
    prt_version = version("petitRADTRANS")
    file_name = f"target__R-{args.resolution}__pRT-{prt_version}.hdf"
    file_path = output_dir / file_name
    save_to_hdf(
        file_path=file_path,
        wlen=wlen.reshape(1, -1),
        flux=flux.reshape(1, -1),
        theta=THETA_0.reshape(1, -1),
    )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
