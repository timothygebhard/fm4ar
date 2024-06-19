"""
Unit tests for `fm4ar.utils.resampling`.
"""

import numpy as np

from fm4ar.utils.resampling import resample_spectrum


def test__resample_spectrum() -> None:
    """
    Test `fm4ar.utils.resampling.resample_spectrum`.
    """

    rng = np.random.default_rng(42)

    # Case 1: NaN at the edges
    old_wlen = np.linspace(1, 2, 101)
    old_flux = rng.standard_normal(101)
    new_wlen = np.linspace(1, 2, 51)
    new_wlen, new_flux = resample_spectrum(new_wlen, old_wlen, old_flux)
    assert new_wlen.shape == new_flux.shape
    assert np.all(np.isfinite(new_flux))
    assert len(new_wlen) == 49

    # Case 1: No NaN at the edges
    old_wlen = np.linspace(1, 2, 101)
    old_flux = rng.standard_normal(101)
    new_wlen = np.linspace(1.2, 1.8, 51)
    new_wlen, new_flux = resample_spectrum(new_wlen, old_wlen, old_flux)
    assert new_wlen.shape == new_flux.shape
    assert np.all(np.isfinite(new_flux))
    assert len(new_wlen) == 51

    # Case 2: With uncertainties
    old_wlen = np.linspace(1, 2, 101)
    old_flux = rng.standard_normal(101)
    old_errs = rng.standard_normal(101)
    new_wlen = np.linspace(1.2, 1.8, 51)
    new_wlen, new_flux, new_errs = resample_spectrum(
        new_wlen, old_wlen, old_flux, old_errs
    )
    assert new_wlen.shape == new_flux.shape == new_errs.shape
    assert np.all(np.isfinite(new_flux))
    assert np.all(np.isfinite(new_errs))
    assert len(new_wlen) == 51
