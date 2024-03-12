"""
Define a parser for prior configurations.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PriorConfig(BaseModel):
    """
    Configuration for the prior distribution.
    """

    dataset: Literal["vasist_2023"] = Field(
        default="vasist_2023",
        description="Name of the dataset whose prior distribution we use.",
    )
    parameters: dict[str, str] = Field(
        default={},
        description=(
            "Mapping of parameter names to actions. This can be empty in "
            "cases where we do not run nested sampling, but only want to "
            "use it, e.g., to evaluate prior value for a given `theta`."
        ),
    )
    random_seed: int = Field(
        default=42,
        description="Random seed to use for the prior distribution",
    )
