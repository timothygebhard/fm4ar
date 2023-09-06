"""
Unit tests for `fm4ar.utils.resampling`.
"""

import numpy as np

from fm4ar.utils.resampling import resample_spectrum


def test__resample_spectrum() -> None:
    """
    Test `fm4ar.utils.resampling.resample_spectrum`.
    """

    # Case 1: NaN at the edges
    old_wlen = np.linspace(1, 2, 101)
    old_flux = np.random.randn(101)
    new_wlen = np.linspace(1, 2, 51)
    new_wlen, new_flux = resample_spectrum(new_wlen, old_wlen, old_flux)
    assert new_wlen.shape == new_flux.shape
    assert np.all(np.isfinite(new_flux))
    assert len(new_wlen) == 49

    # Case 1: No NaN at the edges
    old_wlen = np.linspace(1, 2, 101)
    old_flux = np.random.randn(101)
    new_wlen = np.linspace(1.2, 1.8, 51)
    new_wlen, new_flux = resample_spectrum(new_wlen, old_wlen, old_flux)
    assert new_wlen.shape == new_flux.shape
    assert np.all(np.isfinite(new_flux))
    assert len(new_wlen) == 51
