"""
Test that we can construct a flow matching model from a configuration.
"""

from pathlib import Path
from shutil import copyfile

import h5py
import numpy as np
import pytest
import torch
from deepdiff import DeepDiff

from fm4ar.models.build_model import build_model
from fm4ar.training.preparation import prepare_new, prepare_resume
from fm4ar.training.stages import StageConfig, initialize_stage, train_stages
from fm4ar.training.train_validate import (
    move_batch_to_device,
    train_epoch,
    validate_epoch,
)
from fm4ar.utils.config import load_config
from fm4ar.utils.paths import get_experiments_dir
from fm4ar.utils.tracking import RuntimeLimits

# Define some constants for the mock data
N_TOTAL = 22  # number of samples in the mock dataset
BATCH_SIZE = 5  # batch size for the mock data
DIM_THETA = 16  # required to use `vasist_2023` feature scaler
N_BINS = 39  # number of bins in the mock data


@pytest.fixture
def experiment_dir(tmp_path: Path) -> Path:
    """
    Create a dummy experiment directory and return the path to it.
    """

    # Create the dummy experiment directory
    experiment_dir = tmp_path / "dummy_experiment_npe"
    experiment_dir.mkdir()

    # Copy over the template configuration
    template_dir = get_experiments_dir() / "templates" / "npe"
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
    rng = np.random.default_rng(42)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("theta", data=rng.random((N_TOTAL, DIM_THETA)))
        f.create_dataset("wlen", data=rng.random((N_TOTAL, N_BINS)))
        f.create_dataset("flux", data=rng.random((N_TOTAL, N_BINS)))

    return file_path


# We run for both flow libraries
@pytest.mark.parametrize(
    "flow_wrapper_config, random_seed, expected_sum, expected_loss",
    [
        (
            {
                "flow_library": "glasflow",
                "kwargs": {
                    "num_flow_steps": 3,
                    "base_transform_type": "rq-coupling",
                    "base_transform_kwargs": {
                        "hidden_dim": 64,
                        "num_transform_blocks": 2,
                        "activation": "ELU",
                        "dropout_probability": 0.1,
                        "use_batch_norm": True,
                        "num_bins": 10,
                    },
                },
            },
            0,
            868.3368530273438,
            45.32202657063802,
        ),
        (
            {
                "flow_library": "normflows",
                "kwargs": {
                    "num_flow_steps": 3,
                    "base_transform_type": "rq-coupling",
                    "base_transform_kwargs": {
                        "num_blocks": 2,
                        "num_hidden_channels": 64,
                        "num_bins": 10,
                        "tail_bound": 10,
                        "activation": "ELU",
                        "dropout_probability": 0.1,
                    },
                },
            },
            1,
            200.9569854736328,
            32.757826487223305,
        ),
    ],
)
@pytest.mark.integration_test
def test__npe_model(
    flow_wrapper_config: dict,
    random_seed: int,
    expected_sum: float,
    expected_loss: float,
    experiment_dir: Path,
    path_to_dummy_dataset: Path,
) -> None:
    """
    Integration test to check that we can build an NPE model from the
    template configuration and send some mock data through it.
    """

    # Set the random seed for reproducibility
    torch.manual_seed(123456)

    # Read in template configuration (which was copied to the experiment dir)
    config = load_config(experiment_dir=experiment_dir)
    config["model"]["random_seed"] = random_seed

    # Overwrite the dataset section
    # This should give us 3 training batches and 1 validation batch
    config["dataset"]["file_path"] = path_to_dummy_dataset.as_posix()
    config["dataset"]["n_train_samples"] = 15
    config["dataset"]["n_valid_samples"] = 5

    # Overwrite the the flow_wrapper configuration
    config["model"]["flow_wrapper"] = flow_wrapper_config

    # Prepare the model and the dataset
    model, dataset = prepare_new(experiment_dir=experiment_dir, config=config)

    # Check that the weight initialization is deterministic
    actual_sum = float(sum(p.sum() for p in model.network.parameters()))
    assert np.isclose(actual_sum, expected_sum)

    # Select the first stage; make sure config is suitable for testing
    stage_config = StageConfig(**list(config["training"].values())[0])
    stage_config.batch_size = BATCH_SIZE
    stage_config.logprob_evaluation.interval = 3
    stage_config.use_amp = False

    # Initialize the stage
    train_loader, valid_loader = initialize_stage(
        model=model,
        dataset=dataset,
        resume=False,
        stage_name=list(config["training"].keys())[0],
        stage_config=stage_config,
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
    train_loss = 0.0
    for epoch in range(1, 3):

        # Manually set the model epoch
        model.epoch = epoch
        model.stage_epoch = epoch

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
            assert avg_log_prob is not None and np.isfinite(avg_log_prob)
        if epoch == 2:
            assert avg_log_prob is None

    # Check that the number of epochs and stage name are correct
    assert model.epoch == 2
    assert model.stage_name == "stage_0"
    assert model.stage_epoch == 2

    # Check the last train loss --- this should also be reproducible
    assert np.isclose(train_loss, expected_loss)

    # Check that we can train for two more epochs using the .train() method
    model.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        runtime_limits=RuntimeLimits(max_epochs=4),
        stage_config=stage_config,
    )
    assert model.epoch == 4
    assert model.stage_epoch == 4

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
    for key in model.network.state_dict():
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

    # Check that we can use train_stage()
    done = train_stages(model=model, dataset=dataset)
    assert done
    assert model.epoch == 5

    # Finally, test warning for invalid save_model() call
    model.experiment_dir = None
    with pytest.warns(UserWarning) as user_warning:
        model.save_model()
    assert "no directory was specified" in str(user_warning.list[0].message)
