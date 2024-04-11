"""
Define an abstract base class for simulators.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseSimulator(ABC):
    """
    Common base class for all simulators.
    """

    @abstractmethod
    def __call__(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Run the simulation and return the result.

        Every simulator should take an array of parameters `theta`
        as its input, and return a tuple (wavelengths, fluxes) as
        its result (or None, in case the simulation failed).
        """

        raise NotImplementedError  # pragma: no cover
