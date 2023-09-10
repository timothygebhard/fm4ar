"""
Load a dataset from the given experiment configuration.
Note: This function is defined here to avoid circular imports.
"""

from copy import deepcopy

from fm4ar.datasets.dataset import ArDataset
from fm4ar.datasets.ardevol_martinez_2022 import (
    load_ardevol_martinez_2022_dataset,
)
from fm4ar.datasets.goyal_2020 import load_goyal_2020_dataset
from fm4ar.datasets.vasist_2023 import load_vasist_2023_dataset


def load_dataset(config: dict) -> ArDataset:
    """
    Load a dataset from the given experiment configuration.
    """

    # Do not modify the original configuration when calling pop()
    config = deepcopy(config)

    match (name := config["data"].pop("name")):
        case "ardevol-martinez-2022":
            return load_ardevol_martinez_2022_dataset(config)
        case "goyal-2020":
            return load_goyal_2020_dataset(config)
        case "vasist-2023":
            return load_vasist_2023_dataset(config)
        case _:
            raise ValueError(f"Unknown dataset: `{name}`")
