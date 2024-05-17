"""
This script evaluates the densities of posterior samples from nested
sampling under a given ML model. If the model is "good" and nested
sampling also worked as expected, all samples should have a finite
density.
Note: This script probably needs a GPU to run in a reasonable time.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dynesty.utils import resample_equal
from tqdm import tqdm

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.models.build_model import build_model
from fm4ar.models.npe import NPEModel
from fm4ar.nested_sampling.config import load_config as load_ns_config
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.priors import get_prior
from fm4ar.simulators import get_simulator
from fm4ar.utils.config import load_config as load_ml_config
from fm4ar.utils.hdf import load_from_hdf


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the (machine learning) experiment directory."
    )
    parser.add_argument(
        "--importance-sampling-run",
        type=Path,
        required=True,
        help="Path to the importance sampling run again which to compare."
    )
    parser.add_argument(
        "--nested-sampling-dir",
        type=Path,
        required=True,
        help="Path to the nested sampling directory."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    script_start = time.time()
    print("\nEVALUATE NESTED SAMPLING DENSITIES UNDER ML MODEL\n", flush=True)

    # Get the command line arguments and load nested sampling config
    args = get_cli_arguments()
    ml_config = load_ml_config(experiment_dir=args.experiment_dir)
    ns_config = load_ns_config(experiment_dir=args.nested_sampling_dir)
    sigma = ns_config.likelihood.sigma

    # Load the nested sampling posterior, resample to equal weights, and
    # apply theta scaler (model expects normalized inputs)
    print("Loading nested sampling posterior samples...", end=" ", flush=True)
    samples, weights = load_posterior(args.nested_sampling_dir)
    samples = resample_equal(samples=samples, weights=weights)
    theta_scaler = get_theta_scaler(config=ml_config["theta_scaler"])
    samples = theta_scaler.forward_array(samples)
    print(f"Done! ({len(samples):,} samples)")

    # Load the importance sampling results
    print("Loading importance sampling results...", end=" ", flush=True)
    is_results = load_from_hdf(
        file_path=(
            args.importance_sampling_run
            / "importance_sampling_results.hdf"
        ),
        keys=["log_probs", "weights"],
    )
    print(f"Done! ({len(is_results['log_probs']):,} samples)")

    # Simulate the target spectrum for the nested sampling run
    print("Simulating nested sampling target spectrum...", end=" ", flush=True)
    prior = get_prior(config=ns_config.prior)
    simulator = get_simulator(config=ns_config.simulator)
    theta_obs = np.array([ns_config.ground_truth[n] for n in prior.names])
    if (result := simulator(theta_obs)) is None:
        raise RuntimeError("Failed to simulate the target spectrum!")
    wlen, flux = result
    n_bins = len(wlen)
    print("Done!")

    # Construct the context for the model
    print("Constructing context for the model...", end=" ", flush=True)
    context = {
        "flux": torch.from_numpy(flux).float().unsqueeze(0),
        "wlen": torch.from_numpy(wlen).float().unsqueeze(0),
        "error_bars": sigma * torch.ones(1, n_bins).float(),
    }
    print("Done!")

    # Load the model from best checkpoint
    print("Loading trained model...", end=" ", flush=True)
    model = build_model(
        experiment_dir=args.experiment_dir,
        file_path=args.experiment_dir / "model__best.pt",
        device="auto",
    )
    model.network.eval()
    print("Done!")

    # Set tolerance of ODE solver (for FMPE models)
    model_kwargs = {} if isinstance(model, NPEModel) else {"tolerance": 1e-4}

    # Define chunk sizes
    n_samples = samples.shape[0]
    chunk_size = 8192 if isinstance(model, NPEModel) else 2048
    chunk_sizes = np.diff(np.r_[0:n_samples:chunk_size, n_samples])

    # Evaluate the log-density of the samples (in chunks)
    print("Evaluating log-densities nestes sampling samples:", flush=True)
    log_prob_chunks = []
    with torch.no_grad():
        for i in tqdm(range(len(chunk_sizes)), ncols=80):

            # Select the samples for this chunk
            chunk_start = np.sum(chunk_sizes[:i])
            chunk_end = chunk_start + chunk_sizes[i]
            theta = torch.from_numpy(samples[chunk_start:chunk_end]).float()
            n = theta.shape[0]

            # Adjust the size of the context so that the batch size matches
            # the desired chunk size, and move it to the correct device
            chunk_context = {
                k: v.repeat(n, 1).to(model.device, non_blocking=True)
                for k, v in context.items()
            }

            # Draw samples and corresponding log-probs from the model
            chunk = model.log_prob_batch(
                theta=theta.to(model.device, non_blocking=True),
                context=chunk_context,
                **model_kwargs,
            )
            log_prob_chunks.append(chunk.cpu())

    # Concatenate the log-probs and convert to numpy
    log_prob = torch.cat(log_prob_chunks, dim=0).numpy()

    # Create a histogram plot of the log-densities
    print("\nCreating histogram plot of log-densities...", end=" ", flush=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        log_prob,
        bins=100,
        color="C0",
        histtype="step",
        density=True,
        label="Nested sampling",
    )
    ax.hist(
        is_results["log_probs"],
        bins=100,
        color="C1",
        histtype="step",
        density=True,
        label="Importance sampling (proposals)",
    )
    ax.hist(
        is_results["log_probs"],
        weights=is_results["weights"],
        bins=100,
        color="C2",
        histtype="step",
        density=True,
        label="Importance sampling (with weights)",
    )
    ax.set_xlabel("Log-density of nested sampling posterior samples")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    file_path = "/home/tgebhard/log-densities.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.1f} seconds!\n")
