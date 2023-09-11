"""
Load a trained model and evaluate it on the respective test set.
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

from fm4ar.models.build_model import build_model
from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.models.discrete.normalizing_flow import NormalizingFlow
from fm4ar.datasets import load_dataset
from fm4ar.utils.config import load_config


def get_cli_arguments() -> argparse.Namespace:
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
        default=256,
        help="Number of samples to draw from posterior.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for ODE solver (only needed for flow matching).",
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

    # For some reason, this does occasionally crash with an assertion
    # error: "AssertionError: underflow in dt nan"
    # Idea: Use `dopri8` instead of `dopri5` as the solver?
    with (
        torch.autocast(device_type=args.device),
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
    standardize_theta: Callable[[torch.Tensor], torch.Tensor],
    device: Literal["cpu", "cuda"],
    get_logprob: bool = False,
    **model_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from posterior (with or without log probability).
    """

    with (
        torch.autocast(device_type=device),
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
        samples_as_tensor = standardize_theta(samples_as_tensor.cpu())
        samples = samples_as_tensor.numpy().squeeze()

    return samples, logprob


if __name__ == "__main__":

    script_start = time.time()
    print("\nEVALUATE MODEL ON TEST SET\n")

    # Parse arguments and define shortcuts
    args = get_cli_arguments()

    # Load config and update dataset to test set
    print("Loading config...", end=" ")
    config = load_config(experiment_dir=args.experiment_dir)
    config["data"]["which"] = "test"
    config["data"]["add_noise_to_x"] = False
    if args.n_dataset_samples is not None:
        config["data"]["n_samples"] = int(args.n_dataset_samples)
    print("Done!")

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

    # Define shorthand for standardizing data
    def standardize_theta(theta: torch.Tensor) -> torch.Tensor:
        return dataset.standardize(sample=theta, label="theta", inverse=True)

    # Load the model
    print("Loading model...", end=" ")
    file_path = args.experiment_dir / "model__best.pt"
    model = build_model(file_path=file_path, device=args.device)
    model.network.eval()
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
        # Store theta (in original units)
        theta = standardize_theta(theta=theta)
        list_of_thetas.append(theta.numpy())

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
            standardize_theta=standardize_theta,
            device=args.device,
            get_logprob=args.get_logprob,
            **model_kwargs,
        )
        list_of_samples.append(samples)
        list_of_logprob_samples.append(logprob_samples)

    print()

    # Convert lists to numpy arrays
    thetas = np.array(list_of_thetas)
    samples = np.array(list_of_samples)
    logprob_thetas = np.array(list_of_logprob_thetas)
    logprob_samples = np.array(list_of_logprob_samples)

    # Save the results to an HDF file
    print("Saving results...", end=" ")
    file_path = args.experiment_dir / "results_on_test_set.hdf"
    with h5py.File(file_path, "w") as f:
        f.create_dataset(name="theta", data=thetas)
        f.create_dataset(name="samples", data=samples)
        if args.get_logprob:
            f.create_dataset(name="logprob_theta", data=logprob_thetas)
            f.create_dataset(name="logprob_samples", data=logprob_samples)
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
