"""
Methods to handle simulating data with petitRADTRANS.

This implementation is based on the code from Vasist et al. (2023):
https://gitlab.uliege.be/francois.rozet/sbi-ear
"""

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import petitRADTRANS as prt
import petitRADTRANS.retrieval.models as models
import petitRADTRANS.retrieval.parameter as prm
from joblib import Memory
from petitRADTRANS.nat_cst import r_jup_mean

from fm4ar.utils.timeout import timelimit, TimeoutException


MEMORY = Memory(Path.home(), mmap_mode="c", verbose=0)


class Simulator:
    """
    Convenience wrapper around `compute_emission_spectrum()` that
    handles loading (and caching) the pRT object (`atmosphere`).

    Arguments:
        noisy: Whether noise is added to spectra (default: False).
        kwargs: Simulator settings and constants (e.g. planet distance,
            pressures, ...).
    """

    def __init__(
        self,
        R: int = 1000,
        noisy: bool = False,
        time_limit: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.time_limit = time_limit

        # Constants
        default = {
            "D_pl": 41.2925 * prt.nat_cst.pc,
            "pressure_scaling": 10,
            "pressure_simple": 100,
            "pressure_width": 3,
            "scale": 1e16,
        }

        self.constants = {k: kwargs.get(k, v) for k, v in default.items()}
        self.scale = self.constants.pop("scale")

        # Resolution
        self.R = R
        suffix = "" if R == 1000 else f"_R_{R}"

        # Size of the generated spectra
        # Is there a way to compute this a priori instead of hard-coding it?
        match R:
            case 400:
                self.output_size = 379
            case _:
                self.output_size = 947

        # Initialize atmosphere
        self.atmosphere = MEMORY.cache(prt.Radtrans)(
            line_species=[
                f"H2O_HITEMP{suffix}",
                f"CO_all_iso_HITEMP{suffix}",
                f"CH4{suffix}",
                f"NH3{suffix}",
                f"CO2{suffix}",
                f"H2S{suffix}",
                f"VO{suffix}",
                f"TiO_all_Exomol{suffix}",
                f"PH3{suffix}",
                f"Na_allard{suffix}",
                f"K_allard{suffix}",
            ],
            cloud_species=["MgSiO3(c)_cd", "Fe(c)_cd"],
            rayleigh_species=["H2", "He"],
            continuum_opacities=["H2-H2", "H2-He"],
            wlen_bords_micron=[0.95, 2.45],
            do_scat_emis=True,
        )

        # Set up number of atmospheric layers (levels) and pressure grid
        self.n_atmospheric_layers = self.constants["pressure_simple"] + (
            len(self.atmosphere.cloud_species)
            * (self.constants["pressure_scaling"] - 1)
            * self.constants["pressure_width"]
        )
        self.atmosphere.setup_opa_structure(
            np.logspace(-6, 3, self.n_atmospheric_layers)
        )

        # Noise
        self.noisy = noisy
        self.sigma = 1.25754e-17 * self.scale

    def __call__(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:

        try:
            with timelimit(self.time_limit):
                wavelengths, spectrum = compute_emission_spectrum(
                    self.atmosphere, theta, **self.constants
                )
                spectrum = self.process(spectrum)
        except TimeoutException:
            return None

        if self.noisy:
            spectrum += self.sigma * np.random.standard_normal(spectrum.shape)

        return wavelengths, spectrum

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Processes spectra into network-friendly inputs.
        """

        return np.array(x * self.scale)


def compute_emission_spectrum(
    atmosphere: prt.Radtrans,
    theta: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates the emission spectrum of an exoplanet.
    """

    # Note: In pRT version 2.4.8 and earlier, the cloud parameters were called
    # `log_X_cb_<...>` but have since been renamed to `eq_scaling_<...>`. Using
    # the wrong names will give spectra that look *very* different and have the
    # wrong abundances, too.
    names = [
        "C/O",
        "Fe/H",
        "log_pquench",
        "eq_scaling_Fe(c)",
        "eq_scaling_MgSiO3(c)",
        "fsed",
        "log_kzz",
        "sigma_lnorm",
        "log_g",
        "R_pl",
        "T_int",
        "T3",
        "T2",
        "T1",
        "alpha",
        "log_delta",
    ]

    kwargs.update(dict(zip(names, theta, strict=True)))
    kwargs["R_pl"] = kwargs["R_pl"] * r_jup_mean

    parameters = {
        k: prm.Parameter(name=k, value=v, is_free_parameter=False)
        for k, v in kwargs.items()
    }

    wavelengths, spectrum = models.emission_model_diseq(
        atmosphere, parameters, AMR=True
    )

    return wavelengths, spectrum


@partial(np.vectorize, signature="(m),(n)->(n)")
def pt_profile(theta: np.ndarray, pressures: np.ndarray) -> np.ndarray:
    """
    Returns the pressure-temperature profile.

    Note: This function is not actually used in the simulator; it is
    just a copy of some of the code that the simulator uses internally
    and that we need if we want to plot a PT profile.
    """

    CO, FeH, *_, T_int, T3, T2, T1, alpha, log_delta = theta
    T3 = ((3 / 4 * T_int**4 * (0.1 + 2 / 3)) ** (1 / 4)) * (1 - T3)
    T2 = T3 * (1 - T2)
    T1 = T2 * (1 - T1)
    delta = (1e6 * 10 ** (-3 + 5 * log_delta)) ** (-alpha)

    return np.array(
        models.PT_ret_model(
            np.array([T1, T2, T3]),
            delta,
            alpha,
            T_int,
            pressures,
            FeH,
            CO,
        )
    )
