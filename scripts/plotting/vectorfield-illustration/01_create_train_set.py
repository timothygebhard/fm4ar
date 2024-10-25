"""
Create the train set for the vector field illustration, namely samples
from a 5-fold star shape.
"""

import time

import h5py
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm import tqdm

if __name__ == "__main__":

    script_start = time.time()
    print("\nCREATE TRAIN SET FOR VECTOR FIELD ILLUSTRATION\n")

    # -------------------------------------------------------------------------
    # Create a shapely polygon that represents a 5-fold star shape
    # -------------------------------------------------------------------------

    print("Creating star shape...", end=" ", flush=True)

    # Define shape parameters
    n = 5
    r1 = 2.0
    r2 = 1.0

    # Construct points of the star
    points = []
    for k in range(n):
        points += [
            (
                r1 * np.cos(2 * np.pi * k / n + np.pi / 2),
                r1 * np.sin(2 * np.pi * k / n + np.pi / 2)
            ),
            (
                r2 * np.cos(2 * np.pi * (k + 1 / 2) / n + np.pi / 2),
                r2 * np.sin(2 * np.pi * (k + 1 / 2) / n + np.pi / 2),
            ),
        ]

    # Convert point list to a shapely polygon
    points = np.array(points)
    star = Polygon(points)

    print("Done!", flush=True)

    # -------------------------------------------------------------------------
    # Draw samples and keep only those that are inside the star
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(seed=42)
    n_samples = 524_288
    samples = []

    with tqdm(total=n_samples, desc="Drawing samples", ncols=80) as pbar:
        while len(samples) < n_samples:
            p = rng.uniform(-2, 2, (2, 1))
            if star.contains(Point(p)):
                samples.append(p)
                pbar.update(1)

    samples = np.array(samples)

    # -------------------------------------------------------------------------
    # Save the results to an HDF file
    # -------------------------------------------------------------------------

    print("Saving data...", end=" ", flush=True)
    with h5py.File("data.hdf", "w") as f:
        f.create_dataset(name="wlen", data=np.zeros(1))
        f.create_dataset(name="flux", data=np.zeros((len(samples), 1)))
        f.create_dataset(name="theta", data=samples.squeeze())
    print("Done!", flush=True)

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
