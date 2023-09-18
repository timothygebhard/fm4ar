"""
Methods for a simplistic toy dataset that can be used for debugging.
"""

import warnings
from copy import deepcopy

import h5py
import numpy as np
import torch

from fm4ar.datasets.dataset import ArDataset
from fm4ar.utils.paths import get_datasets_dir


def simulate_toy_spectrum(
    params: np.ndarray,
    resolution: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple and fast function to simulate a pseudo-spectrum.

    Args:
        params: Array of parameters (arbitrary length, but >= 4).
        resolution: Resolution of the output "spectra".

    Returns:
        A 2-tuple consisting of the "wavelengths" and the "flux" of
        the pseudo-spectra.
    """

    # Make sure there will be some covariances in the posterior
    p = params.copy()
    p[0] -= (p[1] * 0.5)
    p[1] /= 3
    p[2] += p[3] ** 3

    # Generate wavelengths and coefficients
    wlen = np.linspace(0, 1, resolution)
    rng = np.random.RandomState(42)
    coefs = rng.normal(0, 1, (500, len(params))) @ p

    # Compute the pseudo-flux
    result = 10 * np.mean(
        [a_i * np.cos(i * np.pi * wlen) for i, a_i in enumerate(coefs)],
        axis=0,
    )

    return wlen, result


def get_posterior_samples(
    true_spectrum: np.ndarray,
    true_theta: np.ndarray,
    sigma: float = 0.42,
    n_livepoints: int = 500,
    n_samples: int = 1000,
) -> np.ndarray:
    """
    Run nested sampling to get posterior samples for a given spectrum.

    Args:
        true_spectrum: The "target" spectrum in the likelihood function.
            Note: This should usually be `simulate(true_theta) + noise`!
        true_theta: True parameter values.
        sigma: Uncertainty to use in the likelihood; should match the
            standard deviation of the noise added to the spectrum.
        n_livepoints: Number of livepoints for nested sampling.
        n_samples: Number of samples from the posterior.

    Returns:
        Samples from the posterior.
    """

    # Delay imports and ignore the "Found Intel OpenMP ('libiomp') and LLVM
    # OpenMP ('libomp') loaded at the same time" warnings on macOS...
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from scipy.stats import norm
        from nautilus import Prior, Sampler

    resolution = len(true_spectrum)

    # Define a prior
    prior = Prior()
    for i in range(len(true_theta)):
        prior.add_parameter(f'$p_{i}$', dist=norm(loc=0, scale=1.0))

    # Define likelihood function
    def likelihood(param_dict: dict[str, float]) -> float:
        x = np.array(list(param_dict.values()))
        _, pred_spectrum = simulate_toy_spectrum(x, resolution=resolution)
        return float(
            -0.5 * np.sum(((pred_spectrum - true_spectrum) / sigma) ** 2)
        )

    # Set up the sampler
    sampler = Sampler(
        prior=prior,
        likelihood=likelihood,
        n_live=n_livepoints,
    )

    # Run the sampler (see above why we ignore warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler.run(verbose=False, discard_exploration=True)

    # Select samples from posterior
    samples: np.ndarray
    samples, _, _ = sampler.posterior(equal_weight=True)

    # The `choice()` call is needed to make sure we always get `n_samples`
    # samples, even if the `sampler.posterior()` returns fewer than that
    idx = np.random.choice(a=len(samples), size=n_samples, replace=True)
    samples = samples[idx]

    return samples


def load_toy_dataset(config: dict) -> ArDataset:
    """
    Load the toy dataset.
    """

    # Do not modify the original config
    config = deepcopy(config)

    # Get the subset to load (training or test)
    which = config["data"].pop("which", "train")
    which = "train" if which == "training" else which

    # Load data from HDF file
    dataset_dir = get_datasets_dir() / "toy-dataset"
    file_path = dataset_dir / f"{which}.hdf"
    with h5py.File(file_path, "r") as hdf_file:
        if (n_samples := config["data"].get("n_samples")) is None:
            theta = np.array(hdf_file["theta"])
            x = np.array(hdf_file["spectra"])
            wavelengths = np.array(hdf_file["wavelengths"])
        else:
            theta = np.array(hdf_file["theta"][:n_samples])
            x = np.array(hdf_file["spectra"][:n_samples])
            wavelengths = np.array(hdf_file["wavelengths"])

    # Define noise levels
    noise_levels = 0.5

    # Get number of parameters
    ndim = theta.shape[1]

    # Create dataset
    return ArDataset(
        theta=torch.from_numpy(theta),
        x=torch.from_numpy(x),
        wavelengths=torch.from_numpy(wavelengths),
        noise_levels=noise_levels,
        names=[f"$p_{i}$" for i in range(ndim)],
        ranges=[(-3, 3) for _ in range(ndim)],
        **config["data"],
    )
