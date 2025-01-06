"""
Everything related to the target spectrum used for importance sampling
or nested sampling.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from pydantic import BaseModel, Field

from fm4ar.utils.paths import expand_env_variables_in_path


class TargetSpectrumConfig(BaseModel):
    """
    Configuration for the target spectrum.
    """

    file_path: Path = Field(
        ...,
        description="Path to the file containing the target spectrum.",
    )
    index: int = Field(
        default=0,
        description="Index of the target spectrum in the file.",
    )


@dataclass
class TargetSpectrum:
    """
    A target spectrum. We use a dataclass instead of a dictionary for
    better type hinting.
    """

    wlen: np.ndarray
    flux: np.ndarray
    error_bars: np.ndarray
    theta: np.ndarray | None = None


def load_target_spectrum(
    file_path: Path,
    index: int = 0,
) -> TargetSpectrum:
    """
    Load a target spectrum from a file.

    Args:
        file_path: Path to the file containing the target spectrum.
        index: Index of the target spectrum to load. Default: 0.
            This may be useful when the file contains multiple target
            spectra, e.g., when using a proper test set.

    Returns:
        A dataclass object containing the wavelengths, flux and error
        bars (i.e., assumed noise level) of the target spectrum, as well
        as the ground truth theta (if available).
    """

    file_path = expand_env_variables_in_path(file_path)

    # Load the target spectrum from the HDF file
    with h5py.File(file_path, "r") as f:

        # Load the target spectrum (wavelength, flux, error bars)
        wlen = np.array(f["wlen"]).astype(np.float32)
        flux = np.atleast_2d(f["flux"])[index].astype(np.float32)
        error_bars = np.atleast_2d(f["error_bars"])[index].astype(np.float32)

        # If available, load the ground truth theta
        # This is of course not available for real observations
        if "theta" in f.keys():
            theta = np.atleast_2d(f["theta"])[index].astype(np.float32)
        else:
            theta = None

    # Return the target spectrum as a dataclass object
    return TargetSpectrum(
        wlen=wlen,
        flux=flux,
        error_bars=error_bars,
        theta=theta,
    )
