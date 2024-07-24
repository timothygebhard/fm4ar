"""
Measure the time for sampling the prior and running the simulator.
"""

import argparse
from pathlib import Path
from time import time

import pandas as pd
from yaml import safe_load

from fm4ar.priors import PriorConfig, get_prior
from fm4ar.simulators import SimulatorConfig, get_simulator

if __name__ == "__main__":

    script_start = time()
    print("\nMEASURE SAMPLING AND SIMULATION TIME\n")

    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default="vasist-2023.yaml",
        help="Path to the configuration file with the simulator setup."
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Load the prior and the simulator
    print("Loading prior and simulator...", end=" ", flush=True)
    prior = get_prior(PriorConfig(**config["prior"]))
    simulator = get_simulator(SimulatorConfig(**config["simulator"]))
    print("Done!", flush=True)

    # Measure the sampling and simulation time
    print("\nMeasuring sampling and simulation time:\n", flush=True)
    times = []
    n = config["n_repeats"]
    for i in range(n + 1):

        # Measure the time for sampling and simulation
        start = time()
        theta = prior.sample()
        result = simulator(theta)
        total = time() - start

        # Store the time (except for the first run, which is a warm-up)
        if i > 0:
            times.append(total)
            print(f"  Run {i:2d}/{n}: {total:.2f} seconds")

    # Construct data frame and print a summary
    df = pd.DataFrame(times)
    avg = df.mean().values[0]
    std = df.std().values[0]
    print(f"\n\nMean time: {avg:.2f} +/- {std:.2f} seconds\n\n")

    # Save the results to a CSV file
    print("Saving results to CSV file...", end=" ", flush=True)
    file_path = "results_" + args.config.stem + ".csv"
    df.to_csv(file_path, index=False)
    print("Done!", flush=True)

    # Print the total runtime
    print(f"\nThis took {time() - script_start:.1f} seconds!\n")
