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


if __name__ == "__main__":

    script_start = time.time()
    print("\nEVALUATE MODEL ON TEST SET\n")

    # Parse arguments
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
        "--n-dataset-samples",
        type=int,
        default=None,
        help="Number of samples from the test set to use.",
    )
    parser.add_argument(
        "--n-posterior-samples",
        type=int,
        default=4096,
        help="Number of samples to draw from posterior.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for ODE solver (only needed for flow matching).",
    )
    args = parser.parse_args()

    # Define shortcuts
    device = args.device
    experiment_dir = args.experiment_dir
    tolerance = args.tolerance
    n_dataset_sample = args.n_dataset_samples

    # Load config and update dataset to test set
    print("Loading config...", end=" ", flush=True)
    config = load_config(experiment_dir=experiment_dir)
    config["data"]["which"] = "test"
    config["data"]["standardize_theta"] = False
    config["data"]["add_noise_to_x"] = False
    print("Done!", flush=True)

    # Limit the number of samples from the test set
    if n_dataset_sample is not None:
        config["data"]["n_samples"] = int(n_dataset_sample)

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

    # Load the model
    print("Loading model...", end=" ", flush=True)
    file_path = experiment_dir / "model_best.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find model at {file_path}!")
    model = build_model(file_path=file_path, device=device)
    model.network.eval()
    print("Done!\n", flush=True)

    # Prepare the values that we want to save later
    list_of_thetas = []
    list_of_samples = []

    # Evaluate the model
    print("Evaluating model:", flush=True)
    for theta, x in tqdm(dataloader, ncols=80):
        with torch.no_grad():

            # Draw samples from the posterior
            x = x.squeeze().repeat(args.n_posterior_samples, 1).to(device)
            if isinstance(model, FlowMatching):
                samples_as_tensor = model.sample_batch(x, tolerance=tolerance)
            else:
                samples_as_tensor = model.sample_batch(x)

            # Map samples back to original units
            samples_as_tensor = dataset.standardize(
                sample=samples_as_tensor.cpu(),
                label="theta",
                inverse=True
            )

        list_of_thetas.append(theta.squeeze().numpy())
        list_of_samples.append(samples_as_tensor.numpy())

    print(flush=True)

    # Convert lists to numpy arrays
    thetas = np.array(list_of_thetas)
    samples = np.array(list_of_samples)

    # Save the results to an HDF file
    print("Saving results...", end=" ", flush=True)
    file_path = args.experiment_dir / "results_on_test_set.hdf"
    with h5py.File(file_path, "w") as hdf_file:
        hdf_file.create_dataset(name="theta", data=thetas)
        hdf_file.create_dataset(name="samples", data=samples)
    print("Done!", flush=True)

    print(f"This took {time.time() - script_start:.2f} seconds!\n", flush=True)
