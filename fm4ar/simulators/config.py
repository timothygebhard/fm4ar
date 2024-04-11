"""
Define a parser for simulator configurations.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SimulatorConfig(BaseModel):
    """
    Configuration for the simulator.
    """

    dataset: Literal["vasist_2023"] = Field(
        default="vasist_2023",
        description="Name of the dataset whose simulator we use.",
    )
    kwargs: dict[str, Any] = Field(
        default={
            "R": 1000,  # spectral resolution R = λ/Δλ
            "time_limit": 20,  # maximum time (in seconds) per simulation
        },
        description="Additional keyword arguments for the simulator.",
    )
