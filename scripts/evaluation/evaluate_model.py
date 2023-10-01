"""
Evaluate a trained model either on the training or test set.

This script can either be run directly on a GPU node, or it can be
invoked using the `--start-submission` flag to prepare a submission
file and launch a new evaluation job on the cluster.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fm4ar.datasets import load_dataset
from fm4ar.models.build_model import build_model
from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.models.discrete.normalizing_flow import NormalizingFlow
from fm4ar.utils.config import load_config
from fm4ar.utils.git_utils import get_git_hash
from fm4ar.utils.hashing import get_sha512sum
from fm4ar.utils.htcondor import (
    CondorSettings,
    check_if_on_login_node,
    condor_submit_bid,
    create_submission_file,
)


def get_cli_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device on which to run everything.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory with config and checkpoint.",
    )
    parser.add_argument(
        "--get-logprob",
        action="store_true",
        help="Whether to compute the log probability of the samples.",
    )
    parser.add_argument(
        "--n-dataset-samples",
        type=int,
        default=None,
        help="Number of samples from the test set to use.",
    )
    parser.add_argument(
        "--n-posterior-samples",
        type=int,
        default=1024,
        help="Number of samples to draw from posterior.",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help=(
            "If this flag is used, the script will prepare the HTCondor "
            "submission file and launch a new job (but not actually run the "
            "evaluation itself)."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for ODE solver (only needed for flow matching).",
    )
    parser.add_argument(
        "--which",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Which dataset to use for evaluation purposes.",
    )
    args = parser.parse_args()

    return args


def get_logprob_of_theta(
    model: FlowMatching | NormalizingFlow,
    theta: torch.Tensor,
    x: torch.Tensor,
    device: Literal["cpu", "cuda"],
    **model_kwargs: Any,
) -> float:
    """
    Compute the log probability of `theta` given `x`.
    """

    model.model.eval()

    # TODO: Computing the log probability with AMP does not seem to work with
    #   the `NormalizingFlow` models from the glasflow package ... ?!
    use_amp = isinstance(model, FlowMatching)

    # For some reason, this does occasionally crash with an assertion
    # error: "AssertionError: underflow in dt nan"
    # Idea: Use `dopri8` instead of `dopri5` as the solver?
    with (
        torch.autocast(device_type=device, enabled=use_amp),
        torch.no_grad(),
    ):
        try:
            logprob_theta = (
                model.log_prob_batch(
                    theta.to(device), x.to(device), **model_kwargs
                )
                .cpu()
                .numpy()
                .squeeze()
            )
        except AssertionError as e:
            print(e)
            logprob_theta = np.nan

    return float(logprob_theta)


def get_samples(
    model: FlowMatching | NormalizingFlow,
    x: torch.Tensor,
    n_samples: int,
    inverse_theta: Callable[[torch.Tensor], torch.Tensor],
    device: Literal["cpu", "cuda"],
    get_logprob: bool = False,
    **model_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from posterior (with or without log probability).
    """

    model.model.eval()

    # TODO: For now, we disable AMP also for posterior sampling when using a
    #   `NormalizingFlow` model, at least until we understand the issue
    use_amp = isinstance(model, FlowMatching)

    with (
        torch.autocast(device_type=device, enabled=use_amp),
        torch.no_grad(),
    ):
        # Prepare input for model
        x = x.tile(n_samples, 1, 1).to(device)

        # Draw samples from the posterior and compute log probability
        if get_logprob:
            (
                samples_as_tensor,
                logprob_as_tensor,
            ) = model.sample_and_log_prob_batch(x, **model_kwargs)
            logprob = logprob_as_tensor.cpu().numpy()
        else:
            samples_as_tensor = model.sample_batch(x, **model_kwargs)
            logprob = np.full(shape=n_samples, fill_value=np.nan)

        # Map samples back to original units
        samples_as_tensor = inverse_theta(samples_as_tensor.cpu())
        samples = samples_as_tensor.numpy().squeeze()

    return samples, logprob


def prepare_submission_file_and_launch_job(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    """
    Create a submission file and launch a new job on the cluster, which
    will run *without* the `--start-submission` flag. This job will then
    run the actual evaluation.
    """

    print("Preparing submission file...", end=" ")

    # Collect arguments for the job: Start with the path to this script,
    # then add all the arguments that we got from the command line
    # TODO: Is there a better way to do this?
    job_arguments = [Path(__file__).resolve().as_posix()]
    for key, value in vars(args).items():
        key = key.replace("_", "-")
        if (
            key == "start-submission"
            or (key == "get-logprob" and not value)
            or value is None
        ):
            continue
        else:
            value = "" if isinstance(value, bool) else str(value)
            job_arguments.append(f"--{key} {value}")

    # Combine condor arguments with the rest of the condor settings
    condor_settings = CondorSettings(**config["local"]["condor"])
    condor_settings.log_file_name = f"evaluate_on_{args.which}"
    condor_settings.arguments = job_arguments

    # Create submission file and submit job
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=args.experiment_dir,
        file_name=f"evaluate_on_{args.which}.sub",
    )

    print("Done!")
    print("Submitting job...", end=" ")
    condor_submit_bid(bid=condor_settings.bid, file_path=file_path)
    print("Done!\n")


def run_evaluation(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    """
    Run the actual evaluation (either on test or training data).
    """

    script_start = time.time()

    # Update args: Default to 1000 samples from the training set
    if args.n_dataset_samples is None and args.which == "train":
        args.n_dataset_samples = 1000

    # Update the experiment configuration
    # TODO: Should we add noise to the input spectra here or not?
    config["data"]["which"] = args.which
    config["data"]["add_noise_to_x"] = False
    if args.n_dataset_samples is not None:
        config["data"]["n_samples"] = int(args.n_dataset_samples)

    # Load the dataset
    print("Loading dataset...", end=" ")
    dataset = load_dataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print("Done!")

    # Load the model
    print("Loading model...", end=" ")
    file_path = args.experiment_dir / "model__best.pt"
    model_hash = get_sha512sum(file_path=file_path)
    model = build_model(file_path=file_path, device=args.device)
    model.model.eval()
    print("Done!\n")

    # Define model-specific keyword arguments
    if isinstance(model, FlowMatching):
        model_kwargs = {"tolerance": args.tolerance}
    else:
        model_kwargs = {}

    # Prepare the values that we want to save later
    list_of_thetas: list[np.ndarray] = []
    list_of_samples: list[np.ndarray] = []
    list_of_logprob_thetas: list[float] = []
    list_of_logprob_samples: list[np.ndarray] = []

    # Evaluate the model
    print("Evaluating model:")
    for theta, x in tqdm(dataloader, ncols=80):

        # Compute log probability of theta
        if args.get_logprob:
            logprob_theta = get_logprob_of_theta(
                model=model,
                theta=theta,
                x=x,
                device=args.device,
                **model_kwargs,
            )
            list_of_logprob_thetas.append(logprob_theta)

        # Draw samples from posterior and store the result
        samples, logprob_samples = get_samples(
            model=model,
            x=x,
            n_samples=args.n_posterior_samples,
            inverse_theta=dataset.standardizer.inverse_theta,
            device=args.device,
            get_logprob=args.get_logprob,
            **model_kwargs,
        )
        list_of_samples.append(samples)
        list_of_logprob_samples.append(logprob_samples)

        # Store theta (in original units)
        theta = dataset.standardizer.inverse_theta(theta)
        list_of_thetas.append(theta.numpy())

    print()

    # Convert lists to numpy arrays
    thetas = np.array(list_of_thetas)
    samples = np.array(list_of_samples)
    logprob_thetas = np.array(list_of_logprob_thetas)
    logprob_samples = np.array(list_of_logprob_samples)

    # Save the results to an HDF file
    print("Saving results...", end=" ")
    file_path = args.experiment_dir / f"results_on_{args.which}_set.hdf"
    with h5py.File(file_path, "w") as f:
        f.attrs["model_hash"] = model_hash
        f.attrs["git_hash"] = get_git_hash()
        f.create_dataset(name="theta", data=thetas)
        f.create_dataset(name="samples", data=samples)
        if args.get_logprob:
            f.create_dataset(name="logprob_theta", data=logprob_thetas)
            f.create_dataset(name="logprob_samples", data=logprob_samples)
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")


if __name__ == "__main__":

    print("\nEVALUATE MODEL\n")

    # Parse arguments and load experiment configuration
    args = get_cli_arguments()
    config = load_config(experiment_dir=args.experiment_dir)

    # Make sure we don't try to run the actual evaluation on the login node
    check_if_on_login_node(start_submission=args.start_submission)

    # Check if we need to prepare a submission file and launch a new job, or
    # if we run the actual evaluation
    if args.start_submission:
        prepare_submission_file_and_launch_job(args=args, config=config)
    else:
        run_evaluation(args=args, config=config)
