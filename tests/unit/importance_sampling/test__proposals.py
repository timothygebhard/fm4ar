"""
Unit tests for `fm4ar.importance_sampling.proposals`.
"""

from argparse import Namespace
from pathlib import Path
from shutil import copyfile

import h5py
import numpy as np
import pytest
import torch

from fm4ar.importance_sampling.config import load_config as load_is_config
from fm4ar.importance_sampling.proposals import draw_proposal_samples
from fm4ar.models.build_model import build_model
from fm4ar.utils.config import load_config as load_npe_config
from fm4ar.unconditional_flow.config import load_config as load_flow_config
from fm4ar.nn.flows import create_unconditional_flow_wrapper
from fm4ar.utils.paths import get_experiments_dir


@pytest.fixture
def file_path_to_target_spectrum(tmp_path: Path) -> Path:
    """
    Create a file with a pseudo target spectrum.
    """

    # Create a file with a pseudo target spectrum
    file_path = tmp_path / "target_spectrum.hdf"
    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            name="wlen",
            data=np.linspace(1, 2, 947),
            dtype=np.float32,
        )
        f.create_dataset(
            name="flux",
            data=np.random.normal(0, 1, (1, 947)) ** 2,
            dtype=np.float32,
        )

    return file_path


def test__draw_proposal_samples__npe(
    tmp_path: Path,
    file_path_to_target_spectrum: Path,
) -> None:
    """
    Test `draw_proposal_samples` for NPE model.
    """

    # Create a temporary directory for the experiment
    experiment_dir = tmp_path / "npe"
    experiment_dir.mkdir()

    # Copy over the template configuration for a NPE model
    template_dir = get_experiments_dir() / "npe-template"
    copyfile(
        template_dir / "config.yaml",
        experiment_dir / "config.yaml",
    )

    # Define some constants
    dim_theta = 16  # because we are using the vasist_2023 theta scaler
    dim_context = 947  # because we are using the vasist_2023 simulator

    # Load and augment the experiment configuration
    npe_config = load_npe_config(experiment_dir=experiment_dir)
    npe_config["model"]["dim_theta"] = dim_theta
    npe_config["model"]["dim_context"] = dim_context

    # Create and save the NPE model
    model = build_model(
        experiment_dir=experiment_dir,
        config=npe_config,
    )
    model.save_model(name="best", save_training_info=False)

    # Copy over the template configuration for an importance sampling run
    template_dir = get_experiments_dir() / "importance-sampling-template"
    copyfile(
        template_dir / "importance_sampling.yaml",
        experiment_dir / "importance_sampling.yaml",
    )

    # Create the command-line arguments
    args = Namespace(
        experiment_dir=experiment_dir,
        job=0,
        n_jobs=1,
        stage=None,
        target_index=0,
    )

    # Load and adjust the importance sampling configuration
    is_config = load_is_config(experiment_dir=experiment_dir)
    is_config.target_spectrum.file_path = file_path_to_target_spectrum
    n_samples = is_config.draw_proposal_samples.n_samples

    # Draw proposal samples
    theta, log_probs = draw_proposal_samples(args=args, config=is_config)

    # Basic sanity checks
    assert theta.shape == (n_samples, dim_theta)
    assert log_probs.shape == (n_samples,)


def test__draw_proposal_samples__unconditional_flow(
    tmp_path: Path,
    file_path_to_target_spectrum: Path,
) -> None:
    """
    Test `draw_proposal_samples` for unconditional flow model.
    """

    # Create a temporary directory for the experiment
    experiment_dir = tmp_path / "npe"
    experiment_dir.mkdir()

    # Copy over the template configuration for a NPE model
    template_dir = get_experiments_dir() / "unconditional-flow-template"
    copyfile(
        template_dir / "config.yaml",
        experiment_dir / "config.yaml",
    )

    # Define constant
    dim_theta = 16  # because we are using the vasist_2023 theta scaler

    # Load and augment the experiment configuration
    flow_config = load_flow_config(experiment_dir=experiment_dir)

    # Create and save the unconditional model
    model = create_unconditional_flow_wrapper(
        dim_theta=dim_theta,
        flow_wrapper_config=flow_config.model.flow_wrapper,
    )
    torch.save(
        {
            "dim_theta": dim_theta,
            "model_state_dict": model.state_dict(),
        },
        experiment_dir / "model__best.pt",
    )

    # Copy over the template configuration for an importance sampling run
    template_dir = get_experiments_dir() / "importance-sampling-template"
    copyfile(
        template_dir / "importance_sampling.yaml",
        experiment_dir / "importance_sampling.yaml",
    )

    # Create the command-line arguments
    args = Namespace(
        experiment_dir=experiment_dir,
        job=0,
        n_jobs=1,
        stage=None,
        target_index=0,
    )

    # Load and adjust the importance sampling configuration
    is_config = load_is_config(experiment_dir=experiment_dir)
    is_config.target_spectrum.file_path = file_path_to_target_spectrum
    n_samples = is_config.draw_proposal_samples.n_samples

    # Draw proposal samples
    theta, log_probs = draw_proposal_samples(args=args, config=is_config)

    # Basic sanity checks
    assert theta.shape == (n_samples, dim_theta)
    assert log_probs.shape == (n_samples,)
