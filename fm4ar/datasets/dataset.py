"""
Wrapper classes for datasets.
"""

from typing import Any, Literal

import torch
from torch.utils.data import Dataset


# Set default data type globally
torch.set_default_dtype(torch.float32)  # type: ignore


class ArDataset(Dataset):
    """
    Base class for for all atmospheric retrieval datasets.
    """

    def __init__(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        n_samples: int | None = None,
        names: list[str] | None = None,
        ranges: list[tuple[float, float]] | None = None,
        noise_levels: float | torch.Tensor | None = None,
        return_wavelengths: bool = False,
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
            n_samples: Number of samples to use. If `None`, all samples
                are used.
            names: Names of the parameters (in order).
            ranges: Ranges of the parameters (in order).
            noise_levels: Noise levels. If this is a float, the same
                noise level is used for all wavelengths. If a tensor,
                the noise level is wavelength-dependent. If `None`,
                no noise is added.
            return_wavelengths: If True, the dataset returns (theta, x),
                where x has two channels (spectrum and wavelengths).
            standardize_theta: If True, standardize the parameters.
            standardize_x: If True, standardize the (context) data.
            add_noise_to_x: If True, add noise to the (context) data.
        """

        super(ArDataset, self).__init__()

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

        # Store other parameters
        self.names = names
        self.ranges = ranges
        self.noise_levels = noise_levels
        self.wavelengths = wavelengths
        self.return_wavelengths = return_wavelengths
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
        """

        if self.noise_levels is None:
            return x

        return x + self.noise_levels * torch.randn(x.shape)

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, idx: Any) -> tuple[torch.Tensor, torch.Tensor]:

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

        # If needed, return the wavelengths as well
        if self.return_wavelengths:
            n = 1 if isinstance(idx, int) else len(idx)
            x = x.reshape(n, -1)
            wlen = self.wavelengths.tile(n, 1)
            x = torch.dstack([x, wlen]).squeeze()

        return theta.float(), x.float()

    @property
    def theta_dim(self) -> int:
        """
        Return the dimensionality of the parameter space.
        """

        return self.theta.shape[1]

    @property
    def context_dim(self) -> tuple[int, ...]:
        """
        Return the dimensionality of the context.
        """

        if self.return_wavelengths:
            return self.x.shape[1], 2
        return (self.x.shape[1], )
