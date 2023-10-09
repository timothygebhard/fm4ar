"""
Define methods to draw a sample from the prior.

This implementation is based on the code from Vasist et al. (2023):
https://gitlab.uliege.be/francois.rozet/sbi-ear
"""

import numpy as np


# Define prior bounds
LOWER: list[float]
UPPER: list[float]
NAMES: list[str]
LABELS: list[str]
LOWER, UPPER, NAMES, LABELS = zip(  # type: ignore
    *[
        [0.1, 1.6, "C/O", r"${\rm C/O}$"],
        [-1.5, 1.5, "Fe/H", r"$\left[{\rm Fe/H}\right]$"],
        [-6.0, 3.0, "P_quench", r"$\log P_{\rm quench}$"],
        [-2.3, 1.0, "log_X_Fe", r"$\log X_{\rm Fe}$"],
        [-2.3, 1.0, "log_X_MgSiO3", r"$\log X_{\rm MgSiO_3}$"],
        [0.0, 10.0, "f_sed", r"$f_{\rm sed}$"],
        [5.0, 13.0, "log_K_zz", r"$\log K_{zz}$"],
        [1.05, 3.0, "sigma_g", r"$\sigma_g$"],
        [2.0, 5.5, "log_g", r"$\log g$"],
        [0.9, 2.0, "R_P", r"$R_P$"],
        [300.0, 2300.0, "T_0", r"$T_0$"],
        [0.0, 1.0, "T_3/T_connect", r"$\frac{T_3}{T_{\rm connect}}$"],
        [0.0, 1.0, "T_2/T_3", r"$\frac{T_2}{T_3}$"],
        [0.0, 1.0, "T_1/T_2", r"$\frac{T_1}{T_2}$"],
        [1.0, 2.0, "alpha", r"$\alpha$"],
        [0.0, 1.0, "log_delta/alpha", r"$\frac{\log \delta}{\alpha}$"],
    ],
    strict=True,
)


# Define theta_0 (or theta*, or theta_obs) from Vasist et al. (2023)
# fmt: off
THETA_0 = np.array(
    [
        0.55,    # C/0
        0.0,     # Fe/H
        -5.0,    # log_P_quench
        -0.86,   # log_X_cb_Fe(c)
        -0.65,   # log_X_cb_MgSiO3(c)
        3.0,     # f_sed
        8.5,     # log_K_zz
        2.0,     # sigma_g
        3.75,    # log_g
        1.0,     # R_P
        1063.6,  # T_0
        0.26,    # T3
        0.29,    # T2
        0.32,    # T1
        1.39,    # alpha
        0.48,    # log_delta
    ]
)
# fmt: on


class Prior:
    """
    Box uniform prior over atmospheric parameters; see Table 1 in
    Vasist et al. (2023).
    """

    def __init__(self, random_seed: int) -> None:
        """
        Initialize class instance.
        """

        # Store random seed and initialize random number generator
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def sample(self) -> np.ndarray:
        """
        Draw a sample from the prior.
        """

        return self.rng.uniform(LOWER, UPPER)
