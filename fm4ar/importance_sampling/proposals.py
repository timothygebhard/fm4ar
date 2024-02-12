"""
Methods for drawing samples from a proposal distribution.
"""

from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

from fm4ar.models.build_model import build_model
from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.datasets.scaling import get_theta_scaler
from fm4ar.nn.flows import create_unconditional_nsf
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.utils.config import load_config as load_ml_config


def draw_proposal_samples(
    args: Namespace,
    context: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """

    Args:
        args:
        context:

    Returns:

    """

    if args.model_type == "ml":

        if context is None:
            raise ValueError("Context must be provided for ML model!")

        print("Running for ML model (FMPE / NPE)!\n")
        theta, probs = draw_samples_from_ml_model(
            context=context,
            experiment_dir=args.experiment_dir,
            n_samples=args.n_samples,
            tolerance=args.tolerance,
        )

    elif args.model_type == "nested_sampling":
        print("Running with KDE on nested sampling posterior!\n")
        theta, probs = draw_samples_from_nested_sampling_with_kde(
            experiment_dir=args.experiment_dir,
            n_samples=args.n_samples,
        )

    elif args.model_type == "unconditional_flow":
        print("Running for unconditional flow model!\n")
        theta, probs = draw_samples_from_unconditional_flow(
            experiment_dir=args.experiment_dir,
            n_samples=args.n_samples,
        )

    else:
        raise ValueError("Unknown model type!")

    return theta, probs


def draw_samples_from_ml_model(
    context: torch.Tensor,
    experiment_dir: Path,
    n_samples: int,
    tolerance: float = 1e-3,
    checkpoint_name: str = "model__best.pt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained ML model (NPE or FMPE) and draw samples from it.

    Args:
        context: Context for the model. Usually, this will be a tensor
            of shape `(1, n_bins, 3)`, where `n_bins` is the number of
            spectral bins, and the `3` corresponds to the the flux, the
            wavelength, and the flux uncertainty.
        experiment_dir: Path to the experiment directory that holds the
            trained model.
        n_samples: Number of samples to draw from the model.
        tolerance: Tolerance for the ODE solver. Only relevant for FMPE.
        checkpoint_name: Name of the checkpoint file to load. Defaults
            to "model__best.pt".
    """

    # Load the trained model
    print("Loading trained model...", end=" ")
    file_path = experiment_dir / checkpoint_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(file_path=file_path, device=device)
    model.model.eval()
    print("Done!")

    # Load experiment config and construct a standardizer for the data
    print("Loading standardizer...", end=" ")
    config = load_ml_config(experiment_dir=experiment_dir)
    theta_scaler = get_theta_scaler(config=config)
    print("Done!\n")

    # Define additional keywords for the model
    if isinstance(model, FlowMatching):
        model_kwargs = dict(tolerance=tolerance)
    else:
        model_kwargs = dict()

    # Draw samples from the model posterior ("proposal distribution").
    # We do this in a chunked fashion to avoid running out of GPU memory.
    print("Drawing samples from the model posterior:", flush=True)
    theta_chunks = []
    probs_chunks = []
    chunk_sizes = np.diff(np.r_[0 : n_samples : 1000, n_samples])
    for chunk_size in tqdm(chunk_sizes, ncols=80):
        with torch.no_grad():
            theta_chunk, log_probs_chunk = model.sample_and_log_prob_batch(
                context=context.repeat(chunk_size, 1, 1).to(device),
                **model_kwargs,  # type: ignore
            )
        theta_chunk = theta_scaler.inverse(theta_chunk.cpu())
        probs_chunk = torch.exp(log_probs_chunk.cpu())
        theta_chunks.append(theta_chunk.cpu())
        probs_chunks.append(probs_chunk.cpu())
    print(flush=True)

    # Combine all chunks into a single array
    theta = torch.cat(theta_chunks, dim=0).numpy()
    probs = torch.cat(probs_chunks, dim=0).numpy().flatten()

    return theta, probs


def draw_samples_from_nested_sampling_with_kde(
    experiment_dir: Path,
    n_samples: int,
    bw_method: str | float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load posterior samples from nested sampling, apply a KDE, and draw
    samples with corresponding probabilities from the KDE.

    This probably only works for simple examples with a small number of
    parameters. For high-dimensional settings, the KDE  will produce a
    bad approximation of the posterior.

    Args:
        experiment_dir: Path to the experiment directory that holds the
            results of a nested sampling run.
        n_samples: Number of samples to draw from the KDE.
        bw_method: Bandwidth method for the KDE. See the documentation
            of `scipy.stats.gaussian_kde` for details.
    """

    # Load the nested sampling posterior
    print("Loading nested sampling posterior...", end=" ")
    ns_samples, ns_weights = load_posterior(experiment_dir=experiment_dir)
    print("Done!\n")

    # Fit the samples with a Gaussian KDE
    print("Fitting samples with Gaussian KDE...", end=" ")
    kde = gaussian_kde(
        dataset=ns_samples.T,
        weights=ns_weights,
        bw_method=bw_method,
    )
    print("Done!")

    # Draw samples from the KDE and compute their probabilities
    # TODO: Maybe this should also be done in a chunked fashion?
    print("Drawing samples from KDE...", end=" ", flush=True)
    theta = kde.resample(size=n_samples).T
    probs = kde.pdf(theta.T)
    print("Done!\n")

    return theta, probs


def draw_samples_from_unconditional_flow(
    experiment_dir: Path,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained unconditional flow model and draw samples from it.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create scaler
    # TODO: Probably this should not be hardcoded!
    config = dict(data=dict(name="vasist-2023", theta_scaler="standardizer"))
    scaler = get_theta_scaler(config=config)

    # Load the unconditional flow model
    print("Loading unconditional flow model...", end=" ")
    model = create_unconditional_nsf()
    model.to(device)
    print("Done!")

    # Load the checkpoint
    print("Loading checkpoint...", end=" ")
    file_path = experiment_dir / "model__best.pt"
    state_dict = torch.load(file_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    print("Done!")

    # Draw samples from the unconditional flow model
    print("Drawing samples from unconditional flow...", end=" ", flush=True)
    model.eval()
    with torch.no_grad():
        samples, logprob = model.sample(num_samples=n_samples)
    theta = scaler.inverse(samples.cpu()).numpy()
    probs = torch.exp(logprob).cpu().numpy()

    print("Done!\n")

    return theta, probs
