"""
Convenience functions for loading simulators from configurations.
"""

from fm4ar.datasets.base_classes import BaseSimulator
from fm4ar.nested_sampling.config import SimulatorConfig


def get_simulator(config: SimulatorConfig) -> BaseSimulator:
    """
    Load a simulator from a configuration object.
    """

    if config.dataset == "vasist_2023":
        from fm4ar.datasets.vasist_2023.simulator import Simulator
        return Simulator(**config.kwargs)

    raise ValueError(f"Unknown simulator dataset: {config.dataset}")
