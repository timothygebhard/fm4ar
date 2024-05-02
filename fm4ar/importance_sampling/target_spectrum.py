"""
Methods for loading a target spectrum.
"""

from pathlib import Path

import h5py
import numpy as np

from fm4ar.utils.paths import expand_env_variables_in_path


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
        A dictionary containing the wavelength and flux of the target
        spectrum, as well as the ground truth theta.
    """

    # TODO: Maybe add support for error bars here, too?

    file_path = expand_env_variables_in_path(file_path)

    target = dict()
    with h5py.File(file_path, "r") as f:
        target["wlen"] = np.array(f["wlen"], dtype=np.float32)
        target["flux"] = np.array(f["flux"][index], dtype=np.float32)
        target["theta"] = np.array(f["theta"][index], dtype=np.float32)

    return target
