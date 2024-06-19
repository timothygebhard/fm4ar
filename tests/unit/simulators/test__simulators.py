"""
Unit tests for `fm4ar.simulators`.
"""

import pytest
from pydantic import ValidationError

from fm4ar.datasets.vasist_2023.simulator import (
    Simulator as Vasist2023Simulator,
)
from fm4ar.simulators import get_simulator
from fm4ar.simulators.base import BaseSimulator
from fm4ar.simulators.config import SimulatorConfig


def test__simulator_config() -> None:
    """
    Test `SimulatorConfig`.
    """

    # Case 1: Valid config
    config = SimulatorConfig(dataset="vasist_2023", kwargs={})
    assert config.dataset == "vasist_2023"
    assert config.kwargs == {}

    # Case 2: Invalid config
    with pytest.raises(ValidationError):
        SimulatorConfig(dataset="unknown", kwargs={})  # type: ignore


def test__get_simulator() -> None:
    """
    Test `get_simulator()`.
    """

    # Case 1
    config = SimulatorConfig(dataset="vasist_2023", kwargs={})
    simulator = get_simulator(config)
    assert isinstance(simulator, BaseSimulator)
    assert isinstance(simulator, Vasist2023Simulator)
