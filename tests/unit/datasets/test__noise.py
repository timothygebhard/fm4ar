"""
Tests for `fm4ar.datasets.noise`.
"""

from pathlib import Path

import numpy as np
import pytest

from fm4ar.datasets.noise import (
    DefaultNoiseGenerator,
    TargetSpectrumNoiseGenerator,
    get_noise_generator,
)
from fm4ar.utils.hdf import save_to_hdf


@pytest.fixture
def path_to_target_spectrum(tmp_path: Path) -> Path:
    """
    Create dummy target spectrum in an HDF file and return the path.
    """

    rng = np.random.default_rng(42)

    # Create a target spectrum
    file_path = tmp_path / "target_spectrum.hdf"
    save_to_hdf(
        file_path=file_path,
        wlen=np.linspace(1.0, 2.0, 101),
        flux=rng.normal(loc=1.0, scale=0.1, size=(10, 101)),
        error_bars=rng.uniform(low=0.05, high=0.15, size=(10, 101)),
    )

    return file_path


def test__default_noise_generator() -> None:
    """
    Test `fm4ar.datasets.noise.DefaultNoiseGenerator`.
    """

    wlen = np.linspace(0.0, 1.0, 10)

    # Case 1: Fixed noise level
    noise_generator = DefaultNoiseGenerator(
        sigma_min=0.5,
        sigma_max=0.5,
        random_seed=42,
    )
    error_bars = noise_generator.sample_error_bars(wlen=wlen)
    assert np.allclose(error_bars, 0.5 * np.ones(10))
    noise = noise_generator.sample_noise(error_bars=error_bars)
    assert np.allclose(np.mean(noise), -0.1390516094475158)

    # Case 2: Random noise level
    noise_generator = DefaultNoiseGenerator(
        sigma_min=0.5,
        sigma_max=1.0,
        random_seed=23,
    )
    error_bars = noise_generator.sample_error_bars(wlen=wlen)
    assert np.all(error_bars >= 0.5)
    assert np.all(error_bars <= 1.0)
    noise = noise_generator.sample_noise(error_bars=error_bars)
    assert np.allclose(np.mean(noise), -0.03235029769558608)

    # Case 3: Invalid noise level
    with pytest.raises(ValueError) as value_error:
        _ = DefaultNoiseGenerator(
            sigma_min=-1.0,
            sigma_max=0.5,
            random_seed=42,
        )
    assert "sigma values must be non-negative!" in str(value_error)


def test__target_spectrum_noise_generator(
    path_to_target_spectrum: Path,
) -> None:
    """
    Test `fm4ar.datasets.noise.TargetSpectrumNoiseGenerator`.
    """

    noise_generator = TargetSpectrumNoiseGenerator(
        file_path=path_to_target_spectrum,
        index=0,
        random_seed=42,
    )

    error_bars = noise_generator.sample_error_bars(
        wlen=np.linspace(1.0, 2.0, 101)
    )
    assert np.isclose(np.mean(error_bars), 0.10351526)

    noise = noise_generator.sample_noise(error_bars=error_bars)
    assert np.isclose(np.mean(noise), -0.004841055177698597)


def test__get_noise_generator(path_to_target_spectrum: Path) -> None:
    """
    Test `fm4ar.datasets.noise.get_noise_generator`.
    """

    # Case 0: Invalid noise generator
    with pytest.raises(ValueError) as value_error:
        _ = get_noise_generator(
            config=dict(
                type="InvalidNoiseGenerator",
                kwargs=dict(),
            )
        )
    assert "Unknown noise generator:" in str(value_error)

    # Case 1: Default noise generator
    noise_generator = get_noise_generator(
        config=dict(
            type="DefaultNoiseGenerator",
            kwargs=dict(
                sigma_min=0.5,
                sigma_max=1.0,
                random_seed=23,
            ),
        )
    )
    assert isinstance(noise_generator, DefaultNoiseGenerator)

    # Case 2: Target spectrum noise generator
    noise_generator = get_noise_generator(
        config=dict(
            type="TargetSpectrumNoiseGenerator",
            kwargs=dict(
                file_path=path_to_target_spectrum,
                index=0,
            ),
        )
    )
    assert isinstance(noise_generator, TargetSpectrumNoiseGenerator)
