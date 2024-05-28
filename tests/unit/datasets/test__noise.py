"""
Tests for `fm4ar.datasets.noise`.
"""

import numpy as np
import pytest

from fm4ar.datasets.noise import DefaultNoiseGenerator, get_noise_generator


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
    assert np.all(0.5 <= error_bars)
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


def test__get_noise_generator() -> None:
    """
    Test `fm4ar.datasets.noise.get_noise_generator`.
    """

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

    # Case 2: Invalid noise generator
    with pytest.raises(ValueError) as value_error:
        _ = get_noise_generator(
            config=dict(
                type="InvalidNoiseGenerator",
                kwargs=dict(),
            )
        )
    assert "Unknown noise generator:" in str(value_error)
