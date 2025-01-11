"""
Define methods to draw a sample from the prior.
"""

import numpy as np
from scipy.stats import uniform

from fm4ar.priors.base import BasePrior

# Define prior bounds
LOWER: tuple[float]
UPPER: tuple[float]
NAMES: tuple[str]
LABELS: tuple[str]
LOWER, UPPER, NAMES, LABELS = zip(
    *[
        [0.1, 1.6, "C/O", r"${\rm C/O}$"],
        [-0.5, 2.5, "Fe/H", r"$\left[{\rm Fe/H}\right]$"],
        [-6.0, 3.0, "log_P_quench", r"$\log P_{\rm quench}$"],
        [-2.3, 1.0, "S_eq_Fe", r"$S_{\rm eq,Fe}$"],
        [-2.3, 1.0, "S_eq_MgSiO3", r"$S_{\rm eq,MgSiO_3}$"],
        [0.0, 10.0, "f_sed", r"$f_{\rm sed}$"],
        [5.0, 13.0, "log_K_zz", r"$\log K_{zz}$"],
        [1.05, 3.0, "sigma_g", r"$\sigma_g$"],
        [2.0, 5.5, "log_g", r"$\log g$"],
        [0.9, 2.0, "R_P", r"$R_P$"],
        [300.0, 2300.0, "T_int", r"$T_{\rm int}$"],
        [0.0, 1.0, "T_3", r"$T_3$"],
        [0.0, 1.0, "T_2", r"$T_2$"],
        [0.0, 1.0, "T_1", r"$T_1$"],
        [1.0, 2.0, "alpha", r"$\alpha$"],
        [0.0, 1.0, "log_delta", "$\log \delta$"],
    ],
    strict=True,
)


class Prior(BasePrior):
    """
    Box uniform prior over atmospheric parameters.
    See Table 1 in Vasist et al. (2023).
    """

    def __init__(self, random_seed: int = 42) -> None:
        """
        Initialize class instance.

        Args:
            random_seed: Random seed to use for reproducibility.
        """

        super().__init__(random_seed=random_seed)

        # Store names and labels for the parameters
        self.names = NAMES
        self.labels = LABELS

        # Store prior bounds as arrays
        self.lower = np.array(LOWER)
        self.upper = np.array(UPPER)

        # Construct the prior distribution.
        # Quote from scipy docs: "In the standard form, the distribution is
        # uniform on [0, 1]. Using the parameters loc and scale, one obtains
        # the uniform distribution on [loc, loc + scale]."
        self.distribution = uniform(
            loc=self.lower,
            scale=self.upper - self.lower,
        )
