"""
Methods for managing the configuration of a nested sampling run.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from yaml import safe_load

from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER, NAMES


class SamplerConfig(BaseModel):
    """
    Configuration for the nested sampling algorithm / sampler.
    """

    which: Literal["nautilus", "dynesty", "multinest"] = Field(
        ...,
        description="Which nested sampling algorithm to use",
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

    resolution: Literal[400, 1000] = Field(
        ...,
        description="Resolution (R = ∆λ/λ) of the simulation",
    )
    time_limit: int = Field(
        ...,
        ge=1,
        description="Maximum runtime (in seconds) for each simulation",
    )


class ParameterConfig(BaseModel):
    """
    Configuration for each parameter.
    """

    true_value: float = Field(
        ...,
        description="Ground truth value of the parameter.",
    )
    action: Literal["condition", "infer", "marginalize"] = Field(
        ...,
        description=(
            "What to do with this parameter during the nested sampling run. "
            "There are three options:\n"
            "  - `condition`: Condition on the true value of the parameter, "
            "       that is, fix it to the true value.\n"
            "  - `infer`: Infer the value of the parameter. This is the "
            "       most common behavior; nested sampling will compute a "
            "       posterior distribution for the parameter.\n"
            "  - `marginalize`: Marginalize over the parameter. This means "
            "        that the parameter will still be randomly sampled from "
            "        the prior, but without handing control to the nested "
            "        sampling algorithm."
        ),
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

    sampler: SamplerConfig
    simulator: SimulatorConfig
    parameters: dict[str, ParameterConfig]
    htcondor: HTCondorConfig

    @model_validator(mode="after")
    def ensure_all_true_values_are_valid(self) -> "Config":
        """
        Make sure that all true values are within the prior bounds.
        """
        for name, parameter in self.parameters.items():
            idx = NAMES.index(name)
            if not (LOWER[idx] <= parameter.true_value <= UPPER[idx]):
                raise ValueError(
                    f"true_value for parameter '{name}' is "
                    f"{parameter.true_value}, which is outside the support "
                    f"of the prior: [{LOWER[idx]}, {UPPER[idx]}]"
                )
        return self

    @model_validator(mode="after")
    def ensure_all_parameters_present(self) -> "Config":
        """
        Make sure that all parameters are present in the prior.
        """
        for name in NAMES:
            if name not in self.parameters:
                raise ValueError(f"Parameter '{name}' not specified.")
        return self

    @model_validator(mode="after")
    def ensure_we_infer_at_least_one_parameter(self) -> "Config":
        """
        Make sure that at least one parameter is inferred.
        """
        if all(
            parameter.action != "infer"
            for parameter in self.parameters.values()
        ):
            raise ValueError("At least one parameter must be set to 'infer'!")
        return self


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
