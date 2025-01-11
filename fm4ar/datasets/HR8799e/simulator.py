"""
Simulate the emission spectrum of an exoplanet using petitRADTRANS.
"""

from pathlib import Path
from typing import Any

import numpy as np
import petitRADTRANS as prt
import petitRADTRANS.retrieval.models as models
import petitRADTRANS.retrieval.parameter as prm
from joblib import Memory
from petitRADTRANS.nat_cst import r_jup_mean

from fm4ar.simulators.base import BaseSimulator
from fm4ar.utils.resampling import resample_spectrum
from fm4ar.utils.timeout import TimeoutException, timelimit

# Cache for the petitRADTRANS atmosphere object
# Using this cache will significantly speed up the creation of the simulator
MEMORY = Memory(Path.home(), mmap_mode="c", verbose=0)


class Simulator(BaseSimulator):
    """
    Convenience wrapper around `compute_emission_spectrum()` that
    handles loading (and caching) the pRT object (`atmosphere`).
    """

    def __init__(
        self,
        time_limit: int = 15,
        **kwargs: Any,
    ) -> None:
        """
        Initialise a new `Simulator` object.

        Arguments:
            time_limit: Maximum time (in seconds) to spend generating a
                single spectrum. If the computation takes longer than this,
                the simulator will return `None`.
            kwargs: Simulator settings and constants (e.g. planet distance,
                pressures, ...).
        """

        super().__init__()

        # Load the wavelength grids for the different instruments
        file_path = Path(__file__).parent / "wlen.npy"
        data = np.load(file_path, allow_pickle=True)
        self.wlen = {
            "wlen": data.item().get("wlen"),
            "idx": data.item().get("idx"),
            "CHARIS": data.item().get("CHARIS"),
            "GPI": data.item().get("GPI"),
            "SPHERE": data.item().get("SPHERE"),
            "GRAVITY": data.item().get("GRAVITY"),
        }

        # The time limit really does not work unless it is an integer
        self.time_limit = int(time_limit)

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

        # Initialize atmosphere
        suffix = ""
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
            wlen_bords_micron=[0.90, 2.50],
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

    def __call__(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:

        # Run the simulator itself
        try:
            with timelimit(self.time_limit):
                wlen, flux = compute_emission_spectrum(
                    self.atmosphere, theta, **self.constants
                )
                flux = self.process(flux)
        except TimeoutException:
            return None

        # Rebin the spectrum to the desired resolution, if possible
        # This is needed because for some values of theta, the spectrum will
        # contain NaNs, which will cause the rebinning to fail. For IS, the
        # NaNs are handled in the `process_theta()` function of the script
        # `run_importance_sampling.py`.
        if any(np.isnan(flux)):
            wlen = self.wlen["wlen"]
            flux = np.full_like(wlen, np.nan)
        else:
            wlen, flux = self.rebin(wlen, flux)

        return wlen, flux

    def rebin(
        self,
        wlen: np.ndarray,
        flux: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rebins the spectrum to the desired resolution.
        """

        # Rebin this to the four different instruments
        _, charis = resample_spectrum(self.wlen["CHARIS"], wlen, flux)
        _, gpi = resample_spectrum(self.wlen["GPI"], wlen, flux)
        _, sphere = resample_spectrum(self.wlen["SPHERE"], wlen, flux)
        _, gravity = resample_spectrum(self.wlen["GRAVITY"], wlen, flux)

        # Concatenate the spectra and sort them by wavelength
        flux = np.concatenate([charis, gpi, sphere, gravity])
        flux = flux[self.wlen["idx"]]

        return self.wlen["wlen"], flux

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

    wlen, flux = models.emission_model_diseq(atmosphere, parameters, AMR=True)

    return wlen, flux
