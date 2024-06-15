"""
Define a parser for importance sampling configurations, and a method
to load such a configuration from a YAML file.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from yaml import safe_load

from fm4ar.priors.config import PriorConfig
from fm4ar.simulators.config import SimulatorConfig
from fm4ar.target_spectrum import TargetSpectrumConfig
from fm4ar.utils.htcondor import HTCondorConfig


class DrawProposalSamplesConfig(BaseModel):
    """
    Configuration for the "draw proposal samples" stage.
    """

    chunk_size: int = Field(
        default=1024,
        ge=1,
        description="Number of proposal samples to draw at once.",
    )
    n_samples: int = Field(
        ...,
        ge=1,
        description="Number of proposal samples to draw.",
    )
    use_amp: bool = Field(
        default=False,
        description="Use AMP for the proposal sampling?",
    )
    htcondor: HTCondorConfig


class MergeProposalSamplesConfig(BaseModel):
    """
    Configuration for the "merge proposal samples" stage.
    """

    delete_after_merge: bool = Field(
        default=True,
        description="Delete proposal sample HDF files after merge?",
    )
    htcondor: HTCondorConfig


class SimulateSpectraConfig(BaseModel):
    """
    Configuration for the "simulate spectra" stage.
    """

    htcondor: HTCondorConfig


class MergeSimulationResultsConfig(BaseModel):
    """
    Configuration for the "merge simulation results" stage.
    """

    delete_after_merge: bool = Field(
        default=True,
        description="Delete simulation result HDF files after merge?",
    )
    htcondor: HTCondorConfig


class ImportanceSamplingConfig(BaseModel):
    """
    Full configuration for an importance sampling run.
    """

    model_config = ConfigDict(protected_namespaces=())

    # General settings
    checkpoint_file_name: str = Field(
        default="model__best.pt",
        description="Name of the model checkpoint file to use.",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed to use for the importance sampling run",
    )
    model_kwargs: dict[str, Any] = Field(
        default={},
        description=(
            "Additional keyword arguments for the posterior model. "
            "Usually, this should only be necessary for FMPE models to "
            "control the settings of the ODE solver (e.g., `tolerance`)."
        ),
    )

    # Target spectrum, prior, and simulator
    target_spectrum: TargetSpectrumConfig
    prior: PriorConfig
    simulator: SimulatorConfig

    # Configuration for the individual stages
    draw_proposal_samples: DrawProposalSamplesConfig
    merge_proposal_samples: MergeProposalSamplesConfig
    simulate_spectra: SimulateSpectraConfig
    merge_simulation_results: MergeSimulationResultsConfig


def load_config(
    experiment_dir: Path,
    name: str = "importance_sampling.yaml",
) -> ImportanceSamplingConfig:
    """
    Load the configuration inside the given experiment directory.
    """

    # Load the configuration file
    config_file = experiment_dir / name
    with open(config_file, "r") as file:
        config_dict = safe_load(file)

    # Construct the configuration object
    return ImportanceSamplingConfig(**config_dict)
