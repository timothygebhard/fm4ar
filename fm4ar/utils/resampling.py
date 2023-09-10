"""
Methods for resampling spectra.
"""

import contextlib
import io
from typing import overload

import numpy as np
from spectres import spectres_numba


@overload
def resample_spectrum(
    new_wlen: np.ndarray,
    old_wlen: np.ndarray,
    old_flux: np.ndarray,
    old_errs: None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ...


@overload
def resample_spectrum(
    new_wlen: np.ndarray,
    old_wlen: np.ndarray,
    old_flux: np.ndarray,
    old_errs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


def resample_spectrum(
    new_wlen: np.ndarray,
    old_wlen: np.ndarray,
    old_flux: np.ndarray,
    old_errs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        old_errs: Old flux uncertainties.

    Returns:
        The resampled wavelengths and flux arrays.
    """

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        output = spectres_numba(
            new_wavs=new_wlen,
            spec_wavs=old_wlen,
            spec_fluxes=old_flux,
            spec_errs=old_errs,
            fill=np.nan,
        )

    if old_errs is None:
        new_flux = output
        mask = np.isnan(new_flux)
        return new_wlen[~mask], new_flux[~mask]

    new_flux, new_errs = output
    mask = np.isnan(new_flux)
    return new_wlen[~mask], new_flux[~mask], new_errs[~mask]
