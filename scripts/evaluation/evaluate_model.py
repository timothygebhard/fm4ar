"""
Load a trained model and evaluate it on the respective test set.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fm4ar.models.build_model import build_model
from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.datasets import load_dataset
from fm4ar.utils.config import load_config


def get_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
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
        "--n-rounds",
        type=int,
        default=4,
        help=(
            "How many rounds of sampling to perform. Total number of "
            "samples is n_posterior_samples * n_rounds. This parameter may "
            "be needed for memory reasons."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for ODE solver (only needed for flow matching).",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    script_start = time.time()
    print("\nEVALUATE MODEL ON TEST SET\n")

    # Parse arguments and define shortcuts
    args = get_cli_arguments()
    device = args.device
    experiment_dir = args.experiment_dir
    get_logprob = args.get_logprob
    n_dataset_sample = args.n_dataset_samples
    tolerance = args.tolerance

    # Load config and update dataset to test set
    print("Loading config...", end=" ", flush=True)
    config = load_config(experiment_dir=experiment_dir)
    config["data"]["which"] = "test"
    config["data"]["add_noise_to_x"] = False
    if n_dataset_sample is not None:
        config["data"]["n_samples"] = int(n_dataset_sample)
    print("Done!", flush=True)

    # Load the dataset
    print("Loading dataset...", end=" ", flush=True)
    dataset = load_dataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print("Done!", flush=True)

    # Define shorthand for standardizing data
    def standardize_theta(theta: torch.Tensor) -> torch.Tensor:
        return dataset.standardize(sample=theta, label="theta", inverse=True)

    # Load the model
    print("Loading model...", end=" ", flush=True)
    file_path = experiment_dir / "model__best.pt"
    model = build_model(file_path=file_path, device=device)
    model.network.eval()
    print("Done!\n", flush=True)

    # Define kwargs for the model calls
    model_kwargs = (
        {"tolerance": tolerance} if isinstance(model, FlowMatching) else {}
    )

    # Prepare the values that we want to save later
    list_of_thetas = []
    list_of_samples = []
    list_of_logprob_thetas = []
    list_of_logprob_samples = []

    # Evaluate the model
    print("Evaluating model:", flush=True)
    with torch.autocast(device_type=args.device):
        for theta, x in tqdm(dataloader, ncols=80):
            # Store theta (in original units)
            theta = standardize_theta(theta=theta)
            list_of_thetas.append(theta.numpy())

            # Compute log probability of theta
            if get_logprob:
                with torch.no_grad():
                    logprob_theta = (
                        model.log_prob_batch(
                            theta.to(device), x.to(device), **model_kwargs
                        )
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                list_of_logprob_thetas.append(logprob_theta)

            # Draw samples from the posterior
            samples_as_array = []
            logprob_as_array = []
            for i in range(args.n_rounds):
                with torch.no_grad():
                    # Prepare input for model
                    if config["data"]["return_wavelengths"]:
                        x = x.tile(args.n_posterior_samples, 1, 1).to(device)
                    else:
                        x = x.tile(args.n_posterior_samples, 1).to(device)

                    # Draw samples from the posterior
                    if get_logprob:
                        (
                            samples_as_tensor,
                            logprob_as_tensor,
                        ) = model.sample_and_log_prob_batch(x, **model_kwargs)
                        logprob = logprob_as_tensor.cpu().numpy()
                    else:
                        samples_as_tensor = model.sample_batch(
                            x, **model_kwargs
                        )
                        logprob = np.array([])

                    logprob_as_array.append(logprob)

                    # Map samples back to original units and store
                    samples_as_tensor = standardize_theta(
                        samples_as_tensor.cpu()
                    )
                    samples_as_array.append(
                        samples_as_tensor.numpy().squeeze()
                    )

            # Store samples and log probabilities
            list_of_samples.append(np.concatenate(samples_as_array))
            list_of_logprob_samples.append(np.concatenate(logprob_as_array))

    print(flush=True)

    # Convert lists to numpy arrays
    thetas = np.array(list_of_thetas)
    samples = np.array(list_of_samples)
    logprob_thetas = np.array(list_of_logprob_thetas)
    logprob_samples = np.array(list_of_logprob_samples)

    # Save the results to an HDF file
    print("Saving results...", end=" ", flush=True)
    file_path = args.experiment_dir / "results_on_test_set.hdf"
    with h5py.File(file_path, "w") as f:
        f.create_dataset(name="theta", data=thetas)
        f.create_dataset(name="samples", data=samples)
        if get_logprob:
            f.create_dataset(name="logprob_theta", data=logprob_thetas)
            f.create_dataset(name="logprob_samples", data=logprob_samples)
    print("Done!", flush=True)

    print(f"This took {time.time() - script_start:.2f} seconds!\n", flush=True)
