"""
Measure the time to simulate a single spectrum with pRT at R=1000.
"""

import time

import numpy as np
from cpuinfo import get_cpu_info
from tqdm import tqdm

from fm4ar.datasets.vasist_2023.prior import Prior, THETA_0
from fm4ar.datasets.vasist_2023.simulator import Simulator


if __name__ == "__main__":

    script_start = time.time()
    print("\nMEASURE SIMULATION TIME\n")

    # Print CPU info
    print("CPU:", get_cpu_info()["brand_raw"], "\n")

    # Create the prior
    prior = Prior(random_seed=42)

    # Initialize the simulator and initialize it
    print("Initializing simulator...", end=" ", flush=True)
    simulator = Simulator(R=1000)
    simulator(theta=THETA_0)
    print("Done!\n")

    # Simulate random spectra and measure the time
    print("Measuring runtimes:")
    runtimes = []
    for _ in tqdm(range(100), ncols=80):
        start = time.time()
        simulator(theta=prior.sample())
        runtimes.append(time.time() - start)
    print()

    # Print the results
    print(f"Mean runtime: {np.mean(runtimes):.2f} seconds")
    print(f"Std. runtime: {np.std(runtimes):.2f} seconds")

    print(f"\nThis took {time.time() - script_start:.3f} seconds!\n")
