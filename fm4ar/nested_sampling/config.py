"""
Define a parser for nested sampling configurations, and a method to
load such a configuration from a YAML file.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from yaml import safe_load

from fm4ar.likelihoods.config import LikelihoodConfig
from fm4ar.priors.config import PriorConfig
from fm4ar.simulators.config import SimulatorConfig
from fm4ar.utils.htcondor import HTCondorConfig


class SamplerConfig(BaseModel):
    """
    Configuration for the nested sampling algorithm / sampler.
    """

    library: Literal["nautilus", "dynesty", "multinest"] = Field(
        ...,
        description="Which nested sampling implementation to use.",
    )
    n_livepoints: int = Field(
        ...,
        ge=1,
        description="Number of live points to use in the nested sampling run",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed to use for the nested sampling run",
    )
    sampler_kwargs: dict[str, Any] = Field(
        default={},
        description=(
            "Additional keyword arguments that are passed to the constructor "
            "of the sampler. Can be used, e.g., to switch between 'normal' "
            "and 'dynamic' nested sampling for `dynesty`."
        ),
    )
    run_kwargs: dict[str, Any] = Field(
        default={},
        description=(
            "Additional keyword arguments that are passed to the run() "
            "method of the nested sampling algorithm. Can be used, e.g., "
            "to define the stopping criterion."
        ),
    )


class NestedSamplingConfig(BaseModel):
    """
    Full configuration for a nested sampling run.
    """

    ground_truth: dict[str, float]
    htcondor: HTCondorConfig
    likelihood: LikelihoodConfig
    sampler: SamplerConfig
    simulator: SimulatorConfig
    prior: PriorConfig


def load_config(
    experiment_dir: Path,
    name: str = "config.yaml",
) -> NestedSamplingConfig:
    """
    Load the configuration inside the given experiment directory.
    """

    # Load the configuration file
    config_file = experiment_dir / name
    with open(config_file, "r") as file:
        config_dict = safe_load(file)

    # Construct the configuration object
    return NestedSamplingConfig(**config_dict)
