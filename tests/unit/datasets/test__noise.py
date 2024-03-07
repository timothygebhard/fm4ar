"""
Tests for `fm4ar.datasets.noise`.
"""

import numpy as np
import pytest

from fm4ar.datasets.noise import (
    get_noise_transform_from_string,
    NoiseGenerator,
)


# noinspection PyUnresolvedReferences
def test__get_noise_transform_from_string() -> None:
    """
    Test `fm4ar.datasets.noise.get_noise_transform_from_string`.
    """

    # Case 1: Valid noise transform (1)
    transform = get_noise_transform_from_string("lambda x, y: x + y")
    assert callable(transform) and transform.__name__ == "<lambda>"
    assert transform.__code__.co_code == (lambda x, y: x + y).__code__.co_code

    # Case 2: Valid noise transform (2)
    transform = get_noise_transform_from_string("lambda wlen, error: error")
    assert callable(transform) and transform.__name__ == "<lambda>"
    assert transform.__code__.co_code == (lambda x, y: y).__code__.co_code

    # Case 3: Invalid noise transforms
    with pytest.raises(ValueError) as value_error:
        get_noise_transform_from_string("invalid")
    assert "Invalid transform string" in str(value_error.value)


def test__noise_generator() -> None:
    """
    Test `fm4ar.datasets.noise.NoiseGenerator`.
    """

    wlen = np.linspace(0.0, 1.0, 10)

    # Case 1: Invalid complexity
    with pytest.raises(ValueError) as value_error:
        NoiseGenerator(random_seed=42, complexity=-1)
    assert "Complexity must be >= 0" in str(value_error.value)

    # Case 2: Constant error bars (std = 1.0 fixed)
    generator = NoiseGenerator(
        random_seed=42,
        complexity=0,
        transform=lambda wlen, error: 0.5 * error,
    )
    error_bars = generator.sample_error_bars(wlen=wlen)
    assert error_bars.shape == (10,)
    assert np.allclose(error_bars, 0.5)
    noise = generator.sample_noise(error_bars=error_bars)
    assert noise.shape == (10,)
    assert np.isclose(np.mean(noise), 0.44806111169875623 / 2)

    # Case 3: Constant error bars (random std)
    generator = NoiseGenerator(
        random_seed=42,
        complexity=1,
    )
    error_bars = generator.sample_error_bars(wlen=wlen)
    assert error_bars.shape == (10,)
    assert np.all(error_bars > 0.0)
    assert len(np.unique(error_bars)) == 1  # random but all the same
    assert np.isclose(error_bars[0], 0.3745401188473625)
    noise = generator.sample_noise(error_bars=error_bars)
    assert noise.shape == (10,)
    assert np.isclose(np.mean(noise), -0.14109344381172906)

    # Case 4: Variable error bars (error bars follow random spline)
    generator = NoiseGenerator(
        random_seed=42,
        complexity=5,
    )
    error_bars = generator.sample_error_bars(wlen=wlen)
    assert error_bars.shape == (10,)
    assert np.all(error_bars > 0.0)
    assert len(np.unique(error_bars)) == 10
    assert np.isclose(np.mean(error_bars), 0.623288518583254)
    noise = generator.sample_noise(error_bars=error_bars)
    assert noise.shape == (10,)
    assert np.isclose(np.mean(noise), -0.23255729716136891)
