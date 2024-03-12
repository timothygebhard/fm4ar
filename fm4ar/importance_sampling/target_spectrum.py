"""
Methods for loading a target spectrum.
"""

from pathlib import Path

import h5py
import numpy as np


def load_target_spectrum(
    file_path: Path,
    index: int = 0,
) -> dict[str, np.ndarray]:

    target = dict()
    with h5py.File(file_path, "r") as f:
        target["wlen"] = np.array(f["wlen"], dtype=np.float32)
        target["flux"] = np.array(f["flux"][index], dtype=np.float32)

    return target
