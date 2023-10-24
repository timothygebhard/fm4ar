"""
Wrapper classes for datasets.
"""

from typing import Any, Optional, TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from fm4ar.datasets.standardization import Standardizer

# Prevent circular imports
if TYPE_CHECKING:
    from fm4ar.datasets.scaling import Scaler


class ArDataset(Dataset):
    """
    Base class for for all atmospheric retrieval datasets.
    """

    def __init__(
        self,
        theta: torch.Tensor,
        flux: torch.Tensor,
        wlen: torch.Tensor,
        noise_levels: float | torch.Tensor,
        noise_floor: float = 0.0,
        names: list[str] | None = None,
        ranges: list[tuple[float, float]] | None = None,
        theta_scaler: Optional["Scaler"] = None,
        standardizer: Standardizer | None = None,
        standardize_theta: bool = True,
        standardize_flux: bool = False,
        add_noise_to_flux: bool = True,
    ) -> None:
        """
        Instantiate a dataset.

        Args:
            theta: Atmospheric parameters (e.g., abundances).
                Expected shape: (n_samples, dim_theta).
            flux: Array with the fluxes of the spectra.
                Expected shape: (n_samples, dim_x).
            wlen: Wavelengths to which the `flux` values correspond to.
                For now, we assume the same wavelengths for all spectra.
                Expected shape: (dim_x, ).
            noise_levels: Noise levels. If this is a float, the same
                noise level is used for all wavelengths. If a tensor,
                the noise level is wavelength-dependent.
            noise_floor: Noise floor, that is, the minimum noise that
                is added to the spectra (in ppm). [This is only used
                for the Ardevol Martinez et al. (2022) train dataset.]
            names: Names of the parameters (in order).
            ranges: Ranges of the parameters (in order).
            theta_scaler: Scaler to use for the parameters.
            standardizer: Standardizer to use for standardizing theta
                and the flux. [Uses pre-computed mean and std.] If None
                is passed, a default standardizer with mean=0 and std=1
                is used.
            standardize_theta: If True, standardize the parameters.
            standardize_flux: If True, standardize the flux.
            add_noise_to_flux: If True, add noise to the flux.
        """

        super().__init__()

        # Store arguments
        self.theta = theta.float()
        self.flux = flux.float()
        self.wlen = wlen.float()
        self.noise_levels = noise_levels
        self.noise_floor = noise_floor
        self.names = names
        self.ranges = ranges
        self.add_noise_to_flux = add_noise_to_flux

        # TODO: This allows backward compatibility with old config files.
        #   This should be removed at some point.
        self.theta_scaler = theta_scaler
        self.standardizer = (
            Standardizer() if standardizer is None
            else standardizer
        )
        self.standardize_theta = standardize_theta
        self.standardize_flux = standardize_flux

    @property
    def noise_levels_as_tensor(self) -> torch.Tensor:
        """
        Return the noise levels as a tensor.
        """

        if isinstance(self.noise_levels, float):
            return torch.ones_like(self.wlen) * self.noise_levels
        else:
            return torch.Tensor(self.noise_levels)

    def add_noise(self, flux: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the given `flux` based on the `noise_levels`
        and the `noise_floor`.

        Input shape: (n_bins, )
        Output shape: (n_bins, )
        """

        # Make sure that the noise levels are not smaller than the noise floor
        if self.noise_floor > 0.0:
            noise_levels = self.noise_levels_as_tensor.clone()
            noise_levels[noise_levels < self.noise_floor] = self.noise_floor
        else:
            noise_levels = self.noise_levels_as_tensor

        # Sample noise from a normal distribution
        noise = noise_levels * torch.randn(flux.shape)

        return flux + noise

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """

        return len(self.theta)

    def __getitem__(self, idx: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the `idx`-th sample from the dataset, which consists of
        the parameters `theta` and the `context`, which itself consists
        of the flux values, wavelenghts, and uncertainties.

        Output shape:
            theta: (dim_theta, )
            stacked: (n_bins, n_features = 3)
        """

        # Get the flux and parameters
        flux = self.flux[idx]
        theta = self.theta[idx]

        # Add noise to the spectrum and standardize the parameters
        if self.add_noise_to_flux:
            flux = self.add_noise(flux)

        # TODO: Update this once we no longer need backward compatibility
        # If requested, standardize the flux and the pararameters
        if self.theta_scaler is None:
            if self.standardize_flux:
                flux = self.standardizer.standardize_flux(flux)
            if self.standardize_theta:
                theta = self.standardizer.standardize_theta(theta)
        else:
            theta = self.theta_scaler.forward(theta)

        # Combine flux with wavelengths and noise levels along dim=1.
        # For now, we call this dimension "features".
        # Note: Ideally, we would return a dictionary here, but this makes
        # things more complicated when constructing the embedding networks,
        # and also when caching the contexts (dicts are not hashable).
        context = torch.stack(
            [
                flux,
                self.wlen,
                self.noise_levels_as_tensor,
            ],
            dim=1,
        )

        return theta.float(), context.float()

    @property
    def theta_dim(self) -> int:
        """
        Return the dimensionality of the parameter space.
        """

        return self.theta.shape[1]

    @property
    def context_dim(self) -> tuple[int, int]:
        """
        Return the dimensionality of the context, which consists of the
        flux values, wavelenghts, and uncertainties.
        """

        n_bins = self.flux.shape[1]
        n_features = 3  # x, wavelengths, noise_levels

        return n_bins, n_features
