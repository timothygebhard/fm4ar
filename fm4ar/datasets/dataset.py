"""
Wrapper classes for datasets.
"""

from typing import Any, Literal

import torch
from torch.utils.data import Dataset


class ArDataset(Dataset):
    """
    Base class for for all atmospheric retrieval datasets.
    """

    def __init__(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        noise_levels: float | torch.Tensor,
        n_samples: int | None = None,
        names: list[str] | None = None,
        ranges: list[tuple[float, float]] | None = None,
        standardize_theta: bool = True,
        standardize_x: bool = False,
        add_noise_to_x: bool = False,
        *_: Any,  # Ignore any other arguments
        **__: Any,  # Ignore any other keyword arguments
    ) -> None:
        """
        Instantiate a dataset.

        Args:
            theta: Parameters. Shape: (n_samples, dim_theta).
            x: Data (i.e., a spectrum). Shape: (n_samples, dim_x).
            wavelengths: Wavelengths of the data (i.e., the spectrum).
                For now, we assume that all spectra have the same
                wavelengths. Shape: (dim_x, ).
            noise_levels: Noise levels. If this is a float, the same
                noise level is used for all wavelengths. If a tensor,
                the noise level is wavelength-dependent.
            n_samples: Number of samples to use. If `None`, all samples
                are used.
            names: Names of the parameters (in order).
            ranges: Ranges of the parameters (in order).
            standardize_theta: If True, standardize the parameters.
            standardize_x: If True, standardize the (context) data.
            add_noise_to_x: If True, add noise to the (context) data.
        """

        super().__init__()

        # Select samples
        if n_samples is not None:
            self.theta = theta[:n_samples].float()
            self.x = x[:n_samples].float()
        else:
            self.theta = theta.float()
            self.x = x.float()

        # Compute standardization parameters
        self.standardization = {
            "x": {
                "mean": torch.mean(x, dim=0).float(),
                "std": torch.std(x, dim=0).float(),
            },
            "theta": {
                "mean": torch.mean(theta, dim=0).float(),
                "std": torch.std(theta, dim=0).float(),
            },
        }

        # Store wavelengths and noise levels
        self.wavelengths: torch.Tensor = wavelengths.float()
        self.noise_levels: torch.Tensor = (
            noise_levels if isinstance(noise_levels, torch.Tensor)
            else noise_levels * torch.ones_like(wavelengths)
        ).float()

        # Store other parameters
        self.names = names
        self.ranges = ranges
        self.standardize_theta = standardize_theta
        self.standardize_x = standardize_x
        self.add_noise_to_x = add_noise_to_x

    def standardize(
        self,
        sample: torch.Tensor,
        label: Literal["x", "theta"],
        inverse: bool = False,
    ) -> torch.Tensor:
        """
        Standardize the given `sample` using the parameters for `label`.
        """

        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]

        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the given spectra based on the `noise_levels`.

        Input shape: (n_bins, )
        Output shape: (n_bins, )
        """
        return x + self.noise_levels * torch.randn(x.shape)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.theta)

    def __getitem__(self, idx: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the `idx`-th sample from the dataset.

        Output shape:
            theta: (dim_theta, )
            stacked: (n_bins, n_features)
        """

        # Get the spectrum and parameters
        x = self.x[idx]
        theta = self.theta[idx]

        # Add noise to the spectrum and standardize the parameters
        if self.add_noise_to_x:
            x = self.add_noise(x)

        # Standardize the parameters and the spectrum
        if self.standardize_x:
            x = self.standardize(x, "x")
        if self.standardize_theta:
            theta = self.standardize(theta, "theta")

        # Combine spectra with wavelengths and noise levels along dim=1.
        # For now, we call this dimension "features".
        # Note: Ideally, we would return a dictionary here, but this makes
        # things more complicated when constructing the embedding networks,
        # and also when caching the contexts (dicts are not hashable).
        stacked = torch.stack(
            [
                x,  # Flux values (possibly standardized)
                self.wavelengths,  # Corresponding wavelengths
                self.noise_levels,  # Uncertainties on flux values
            ],
            dim=1,
        )

        return theta.float(), stacked.float()

    @property
    def theta_dim(self) -> int:
        """
        Return the dimensionality of the parameter space.
        """

        return self.theta.shape[1]

    @property
    def context_dim(self) -> tuple[int, int]:
        """
        Return the dimensionality of the (stacked) context.
        """

        n_bins = self.x.shape[1]
        n_features = 3  # x, wavelengths, noise_levels

        return n_bins, n_features
