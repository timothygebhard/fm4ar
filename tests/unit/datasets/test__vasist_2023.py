"""
Tests for the `vasist_2023` module.
"""

import numpy as np

from fm4ar.datasets.vasist_2023.prior import (
    LABELS,
    LOWER,
    NAMES,
    THETA_0,
    UPPER,
    Prior,
)
from fm4ar.datasets.vasist_2023.simulator import Simulator


def test__constants() -> None:
    """
    Test the constants defined in the `vasist_2023` module.
    """

    # Check that everything has the correct length
    assert len(NAMES) == 16
    assert len(LABELS) == 16
    assert len(LOWER) == 16
    assert len(UPPER) == 16

    # Check that everything has the correct type
    assert all(isinstance(name, str) for name in NAMES)
    assert all(isinstance(label, str) for label in LABELS)
    assert all(isinstance(lower, float) for lower in LOWER)
    assert all(isinstance(upper, float) for upper in UPPER)


def test__prior() -> None:
    """
    Test the `Prior` class.
    """

    prior = Prior(random_seed=42)

    # Check that we can deterministically sample from the prior
    sample = prior.sample()
    assert sample.shape == (16,)
    assert np.isclose(np.sum(sample), 1073.7491322968222)

    # Check that we can evaluate the prior
    # It's a uniform prior, so all samples should have the same probability
    assert np.isclose(prior.evaluate(THETA_0), 1.887564381187481e-09)
    assert np.isclose(prior.evaluate(sample), 1.887564381187481e-09)

    # Check that the prior is zero outside the bounds
    assert np.isclose(prior.evaluate(prior.lower - 1), 0.0)
    assert np.isclose(prior.evaluate(prior.upper + 1), 0.0)


def test__simulator() -> None:
    """
    Test the `Simulator` class.
    """

    # Test that we can simulate the benchmark spectrum at R=1000
    simulator = Simulator(random_seed=42, R=1000)
    result = simulator(THETA_0)
    assert result is not None
    wlen, flux = result
    assert wlen.shape == (947,)
    assert flux.shape == (947,)
    assert np.isclose(np.sum(wlen), 1499.0917524698211)
    assert np.isclose(np.sum(flux), 1254.8981023103083)

    # Test that we can limit the runtime of the simulator
    simulator = Simulator(random_seed=42, R=1000, time_limit=1)
    assert simulator(THETA_0) is None

    # Check that the simulation order does not matter
    prior = Prior(random_seed=42)
    x = prior.sample()
    y = prior.sample()
    simulator = Simulator(random_seed=42, R=400)
    result_x_first = simulator(x)
    result_y_second = simulator(y)
    simulator = Simulator(random_seed=42, R=400)
    result_y_first = simulator(y)
    result_x_second = simulator(x)
    assert result_x_first is not None
    assert result_x_second is not None
    assert result_y_first is not None
    assert result_y_second is not None
    assert np.allclose(result_x_first, result_x_second)
    assert np.allclose(result_y_first, result_y_second)
