"""
Everything related to the target spectrum used for importance sampling
or nested sampling.
"""

from pathlib import Path

import h5py
import numpy as np
from pydantic import BaseModel, Field

from fm4ar.utils.paths import expand_env_variables_in_path


class TargetSpectrumConfig(BaseModel):
    """
    Configuration for the target spectrum.
    """

    file_path: Path = Field(
        ...,
        description="Path to the file containing the target spectrum.",
    )
    index: int = Field(
        default=0,
        description="Index of the target spectrum in the file.",
    )


def load_target_spectrum(
    file_path: Path,
    index: int = 0,
) -> dict[str, np.ndarray]:
    """
    Load a target spectrum from a file.

    Args:
        file_path: Path to the file containing the target spectrum.
        index: Index of the target spectrum to load. Default: 0.
            This may be useful when the file contains multiple target
            spectra, e.g., when using a proper test set.

    Returns:
        A dictionary containing the wavelength, flux and error bars
        (i.e., assumed noise level) of the target spectrum, as well as
        the ground truth theta.
    """

    file_path = expand_env_variables_in_path(file_path)

    target = dict()
    with h5py.File(file_path, "r") as f:
        target["wlen"] = np.array(f["wlen"]).flatten()
        target["flux"] = np.array(f["flux"][index]).flatten()
        target["error_bars"] = np.array(f["error_bars"][index]).flatten()
        target["theta"] = np.array(f["theta"][index]).flatten()

    for key, value in target.items():
        target[key] = value.astype(np.float32)

    return target
