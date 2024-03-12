"""
Convenience function for loading simulators from a config.
"""

from fm4ar.simulators.base import BaseSimulator
from fm4ar.simulators.config import SimulatorConfig


def get_simulator(config: SimulatorConfig) -> BaseSimulator:
    """
    Load a simulator from a configuration object.
    """

    if config.dataset == "vasist_2023":
        from fm4ar.datasets.vasist_2023.simulator import Simulator
        return Simulator(**config.kwargs)

    # This should never happen, because the `config` object is validated
    raise ValueError("Unknown simulator dataset!")  # pragma: no cover
