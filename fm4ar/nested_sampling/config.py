"""
Methods for managing the configuration of a nested sampling run.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from yaml import safe_load


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


class LikelihoodConfig(BaseModel):
    """
    Configuration for the likelihood function.
    """

    # TODO: We might want to figure out a way to specify generic distributions
    #  for the likelihood function in the configuration file. For now, we just
    #  assume a multivariate normal distribution.

    # TODO: We need to figure out a way to specify generic covariance matrices
    #  in the configuration file. For now, we just assume that the covariance
    #  matrix is given as `sigma * np.eye(len(x_obs))`.

    sigma: float = Field(
        ...,
        description="Standard deviation of the likelihood function",
    )


class PriorConfig(BaseModel):
    """
    Configuration for the prior distribution.
    """

    dataset: Literal["vasist_2023"] = Field(
        default="vasist_2023",
        description="Name of the dataset whose prior distribution we use.",
    )
    parameters: dict[str, str] = Field(
        ...,
        description="Mapping of parameter names to actions.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed to use for the prior distribution",
    )


class HTCondorConfig(BaseModel):
    """
    Configuration for the HTCondor cluster.
    """

    bid: int = Field(
        ...,
        ge=1,
        le=999,
        description="HTCondor bid",
    )
    n_cpus: int = Field(
        ...,
        ge=1,
        description="Number of CPUs to request",
    )
    memory: int = Field(
        ...,
        ge=1,
        description="Memory (in MB) to request",
    )
    max_runtime: int = Field(
        ...,
        ge=1,
        description="Maximum runtime per job (in seconds)",
    )


class Config(BaseModel):
    """
    Full configuration for a nested sampling run.
    """

    ground_truth: dict[str, float]
    htcondor: HTCondorConfig
    likelihood: LikelihoodConfig
    sampler: SamplerConfig
    simulator: SimulatorConfig
    prior: PriorConfig


def load_config(experiment_dir: Path, name: str = "config.yaml") -> Config:
    """
    Load the configuration inside the given experiment directory.
    """

    # Load the configuration file
    config_file = experiment_dir / name
    with open(config_file, "r") as file:
        config_dict = safe_load(file)

    # Construct the configuration object
    return Config(**config_dict)
