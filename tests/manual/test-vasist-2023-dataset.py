"""
Manual test for the Vasist et al. (2023) dataset.
"""

import sys
import matplotlib.pyplot as plt

from fm4ar.datasets.dataset import load_dataset


if __name__ == "__main__":

    config = {
        "data": {
            "name": "ardevol-martinez-2022",
            "type": "type-2",
            "instrument": "NIRSPEC",
            "which": "test",
            "return_wavelengths": True,
        }
    }
    dataset = load_dataset(config=config)

    theta, x = dataset[0:17]
    print(f"theta.shape = {theta.shape}")
    print(f"x.shape = {x.shape}")
    sys.exit()

    fig, axes = plt.subplots(figsize=(8, 4), nrows=2)

    # Plot different spectra
    axes[0].set_title("Different spectra")
    axes[0].set_xlabel("Wavelength (μm)")
    axes[0].set_ylabel("Transit depth")
    axes[0].set_yscale("log")
    for i in range(8):
        axes[0].plot(dataset.wavelengths, dataset.x[i])

    # Plot different realizations of the same spectrum
    axes[0].set_title("Different noise realizations")
    axes[0].set_xlabel("Wavelength (μm)")
    axes[0].set_ylabel("Transit depth")
    for _ in range(100):
        theta, x = dataset[9]
        axes[1].plot(dataset.wavelengths, x, color="C0", alpha=0.1)
    dataset.noise_levels = 0.0
    axes[1].plot(dataset.wavelengths, dataset.x[9], color="C1")

    plt.tight_layout()
    plt.show()
