"""
Test that we can construct a flow matching model from a configuration.
"""

from shutil import copyfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from deepdiff import DeepDiff

from fm4ar.models.build_model import build_model
from fm4ar.utils.config import load_config
from fm4ar.training.preparation import prepare_new, prepare_resume
from fm4ar.training.stages import initialize_stage, train_stages
from fm4ar.training.train_validate import (
    move_batch_to_device,
    train_epoch,
    validate_epoch,
)
from fm4ar.utils.paths import get_experiments_dir
from fm4ar.utils.tracking import RuntimeLimits


# Define some constants for the mock data
N_TOTAL = 22  # number of samples in the mock dataset
N_LOAD = 20  # number of samples to load from the mock dataset
BATCH_SIZE = 5  # batch size for the mock data
DIM_THETA = 16  # required to use `vasist_2023` feature scaler
N_BINS = 39  # number of bins in the mock data


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    """
    Create a dummy experiment directory and return the path to it.
    """

    # Create the dummy experiment directory
    experiment_dir = tmp_path / "dummy_experiment_fmpe"
    experiment_dir.mkdir()

    # Copy over the template configuration
    template_dir = get_experiments_dir() / "fmpe-template"
    copyfile(
        template_dir / "config.yaml",
        experiment_dir / "config.yaml",
    )

    return experiment_dir


@pytest.fixture
def path_to_dummy_dataset(tmp_path: Path) -> Path:
    """
    Create a dummy dataset for testing and return the path to it.
    """

    file_path = tmp_path / "dummy_dataset.hdf"

    # Create a dummy dataset
    with h5py.File(file_path, "w") as f:
        f.create_dataset("theta", data=np.random.rand(N_TOTAL, DIM_THETA))
        f.create_dataset("wlen", data=np.random.rand(N_TOTAL, N_BINS))
        f.create_dataset("flux", data=np.random.rand(N_TOTAL, N_BINS))

    return file_path


# We run the same test for all combinations of the the *_with_glu flags
@pytest.mark.parametrize(
    "t_theta_with_glu, context_with_glu",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.integration_test
def test__fmpe_model(
    t_theta_with_glu: bool,
    context_with_glu: bool,
    experiment_dir: Path,
    path_to_dummy_dataset: Path,
) -> None:
    """
    Integration test to check that we can build an FMPE model from the
    template confiruation and send some mock data through it.
    """

    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Read in template configuration (which was copied to the experiment dir)
    config = load_config(experiment_dir=experiment_dir)

    # Overwrite the dataset section
    # This should give us 3 training batches and 1 validation batch
    config["dataset"]["file_path"] = path_to_dummy_dataset.as_posix()
    config["dataset"]["train_fraction"] = 0.75
    config["dataset"]["n_samples"] = N_LOAD

    # Set the *_with_glu flags
    config["model"]["t_theta_with_glu"] = t_theta_with_glu
    config["model"]["context_with_glu"] = context_with_glu

    # Prepare the model and the dataset
    model, dataset = prepare_new(experiment_dir=experiment_dir, config=config)

    # Select the first stage; make sure config is suitable for testing
    stage_config = next(iter(config["training"].values()))
    stage_config["batch_size"] = BATCH_SIZE
    stage_config["logprob_epochs"] = 3
    stage_config["early_stopping"] = 0
    stage_config["use_amp"] = False

    # Initialize the stage
    train_loader, valid_loader = initialize_stage(
        model=model,
        dataset=dataset,
        num_workers=0,
        resume=False,
        stage_config=stage_config,
        stage_number=1,
    )

    # Get a batch of mock data
    batch = next(iter(train_loader))
    theta, context = move_batch_to_device(batch, model.device)

    # Send the mock data through the model
    loss = model.loss(theta=theta, context=context)
    assert np.isfinite(loss.item())

    # Check that we can get the context embedding
    expected_embedding_dim = config["model"]["context_embedding_net"][-1][
        "kwargs"
    ]["output_dim"]
    context_embedding = model.network.get_context_embedding(context=context)
    assert context_embedding.shape == (BATCH_SIZE, expected_embedding_dim)

    # Check that we can train and validate manually for two epochs.
    # We train for _two_ epochs to be sure that `validate_epoch()` goes into
    # the branch where we compute the average log probability.
    for epoch in range(1, 3):

        # Manually set the model epoch
        model.epoch = epoch

        # This should be 3 batches
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            stage_config=stage_config,
        )
        assert np.isfinite(train_loss)

        # This should be 1 batch
        avg_log_prob: float | None
        val_loss, avg_log_prob = validate_epoch(
            model=model,
            dataloader=train_loader,
            stage_config=stage_config,
        )
        assert np.isfinite(val_loss)
        if epoch == 1:
            assert avg_log_prob is not None
            assert np.isfinite(avg_log_prob)
        if epoch == 2:
            assert avg_log_prob is None

    # Check that we can train for two more epochs using the .train() method
    model.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        runtime_limits=RuntimeLimits(max_epochs_total=4),
        stage_config=stage_config,
    )

    # Check that we can sample from the model
    samples = model.sample_batch(context=context)
    assert samples.shape == (BATCH_SIZE, DIM_THETA)

    # Check that we can get the log probability of the samples
    log_prob = model.log_prob_batch(theta=samples, context=context)
    assert log_prob.shape == (BATCH_SIZE,)

    # Check that we can sample and get the log probability in one go
    samples, log_prob = model.sample_and_log_prob_batch(context=context)
    assert samples.shape == (BATCH_SIZE, DIM_THETA)
    assert log_prob.shape == (BATCH_SIZE,)

    # Check that we can save the model
    file_path = model.save_model()
    assert file_path is not None and file_path.exists()

    # Check that we can load the model and recover the same configuration
    restored_model = build_model(file_path=file_path)
    assert restored_model.config == model.config
    assert not DeepDiff(
        restored_model.network.state_dict().keys(),
        model.network.state_dict().keys(),
    )
    for key in model.network.state_dict().keys():
        assert torch.allclose(
            model.network.state_dict()[key],
            restored_model.network.state_dict()[key],
        )

    # Check that we can save a snapshot
    snapshot_file_path = model.save_snapshot()
    assert snapshot_file_path is not None and snapshot_file_path.exists()

    # Check that we can use prepare_resume()
    resumed_model, resumed_dataset = prepare_resume(
        experiment_dir=experiment_dir,
        checkpoint_name="model__latest.pt",
        config=config,
    )
    assert resumed_model.config == model.config
    assert np.allclose(dataset.theta, resumed_dataset.theta)

    # Check that we can use train_stages()
    done = train_stages(model=model, dataset=dataset)
    assert done
    assert model.epoch == 4
