"""
Wrapper classes for datasets.
"""

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from fm4ar.datasets.theta_scalers import ThetaScaler, IdentityScaler
from fm4ar.datasets.data_transforms import DataTransform


class SpectraDataset(Dataset):
    """
    Dataset of spectra (and their corresponding simulation parameters).
    """

    def __init__(
        self,
        theta: np.ndarray,
        flux: np.ndarray,
        wlen: np.ndarray,
        theta_scaler: ThetaScaler | None = None,
    ) -> None:
        """
        Instantiate a dataset.

        We start with only the most basic information that we will
        always need, namely the atmospheric parameters `theta` and the
        corresponding simulated (noise-free) spectrum, given by `wlen`
        and `flux`.

        Args:
            theta: Atmospheric parameters (e.g., abundances).
                Expected shape: (n_samples, dim_theta).
            flux: Array with the fluxes of the spectra.
                Expected shape: (n_samples, dim_x).
            wlen: Wavelengths to which the `flux` values correspond to.
                Expected shape(s):
                    - (1, dim_x): Same wavelength for all spectra.
                    - (n_samples, dim_x): Different for each spectrum.
            theta_scaler: Scaler for the parameters `theta`. If None,
                the identity scaler is used (i.e., no scaling).
        """

        super().__init__()

        # Store constructor arguments
        self.theta = theta
        self.flux = flux
        self.wlen = wlen

        # List of transformations that will be applied in __getitem__()
        # Each transform takes a dict (with theta, wlen, flux, and maybe more)
        # and modifies it (e.g., re-binning, sub-sampling, adding noise, ...).
        self.data_transforms: list[DataTransform] = []

        # Scaling transform for the parameters `theta` (e.g., minmax scaling)
        if theta_scaler is not None:
            self.theta_scaler = theta_scaler
        else:
            self.theta_scaler = IdentityScaler()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """

        return len(self.theta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return the `idx`-th sample from the dataset, which is returned
        as a dictionary with the keys "theta", "wlen", and "flux".
        Additional keys (e.g., "error_bars") may be added by the
        transformations in `self.transforms`.

        Note: The tensors in the returned dictionary should *NOT* have
        a batch dimension yet, because the DataLoader will add it later.
        In other words, the shape of the tensors should be (dim_x,) or
        (dim_theta,) and NOT (1, dim_x) or (1, dim_theta).
        """

        # Get the wavelengths, flux and parameters
        wlen = self.wlen[idx] if self.wlen.shape[0] > 1 else self.wlen[0]
        flux = self.flux[idx]
        theta = self.theta[idx]

        # Combine everything into a dict
        # The .copy() here is important, because we will modify the arrays
        # in the data transformations (e.g., add noise), and we don't want
        # to modify the original dataset.
        sample = {
            "theta": theta.copy(),
            "wlen": wlen.copy(),
            "flux": flux.copy(),
        }

        # First apply the data transforms (e.g., adding noise)
        for transform in self.data_transforms:
            sample = transform.forward(sample)

        # Apply the feature scaling for the parameters `theta`
        sample = self.theta_scaler.forward(sample)

        # Convert everything to PyTorch tensors.
        # This step is not a transform because it is non-optional and should
        # always be very the last step, so that all transforms can work with
        # numpy arrays and we only convert to tensors once at the very end.
        sample_as_tensors = {
            key: torch.from_numpy(val).float() for key, val in sample.items()
        }

        # Note: We do NOT move the tensors to the device (GPU) here, because
        # it might break the parallel data loading: The Dataset is consumed
        # by multiple DataLoader workers, but GPU operations usually happen on
        # the main thread. Also, moving the tensors to the GPU here might lead
        # to issues with the automatic memory precision (AMP) training.
        # Instead, we move the tensors to the device in the training loop.

        return sample_as_tensors

    @property
    @lru_cache(maxsize=None)
    def dim_theta(self) -> int:
        """
        Return the number of parameters in the dataset, i.e., the
        dimensionality of `theta`.
        """

        # The dimensionality of the parameters should not be modified by any
        # of the transformations, so we do not need to apply them here.

        return self.theta.shape[1]

    @property
    @lru_cache(maxsize=None)
    def dim_context(self) -> int:
        """
        Return the number of wavelength bins in the spectra, i.e., the
        dimensionality of the context (`flux` and `wlen`).
        """

        # The dimensionality of the spectra can be modified by some of the
        # data transformations, such as subsampling or re-binning. Therefore,
        # we need to infer the number of bins dynamically.

        # Note: Even in these cases, the number of bins must be the same for
        # all spectra (at least within one training stage), otherwise we can't
        # construct batches in the DataLoader.

        return self.__getitem__(0)["flux"].shape[0]
