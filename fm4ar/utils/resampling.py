"""
Methods for resampling spectra.
"""

import contextlib
import io

import numpy as np
from spectres import spectres_numba


def resample_spectrum(
    new_wlen: np.ndarray,
    old_wlen: np.ndarray,
    old_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a spectrum to the given wavelength grid.

    This is a thin wrapper around the `spectres` package that handles
    the `print()`-style warnings and removes the NaNs from the output
    arrays that arise at the edges of the wavelength grid because the
    resampling procedure is flux-preserving.

    Args:
        new_wlen: New wavelength grid.
        old_wlen: Old wavelength grid.
        old_flux: Old flux values.

    Returns:
        The resampled wavelengths and flux arrays.
    """

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        new_flux = spectres_numba(
            new_wlen,
            old_wlen,
            old_flux,
            fill=np.nan,
        )
    mask = np.isnan(new_flux)

    return new_wlen[~mask], new_flux[~mask]
