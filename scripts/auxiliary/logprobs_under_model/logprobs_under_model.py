"""
Evaluate log-probability of a given set of samples under a given model.
"""

import argparse
import time
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fm4ar.datasets.theta_scalers import ThetaScaler, get_theta_scaler
from fm4ar.models.build_model import build_model
from fm4ar.models.fmpe import FMPEModel
from fm4ar.models.npe import NPEModel
from fm4ar.nn.flows import FlowWrapper, create_unconditional_flow_wrapper
from fm4ar.target_spectrum import load_target_spectrum
from fm4ar.utils.config import load_config as load_experiment_config
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the directory with the trained model.",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        required=True,
        help=(
            "Path to the HDF file contained the samples whose log-probability "
            "we want to evaluate under the model. This can be, for example, "
            "the results.hdf file of an IS run."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=(
            "Number of samples to load from the HDF file. If None, all "
            "samples are loaded (default)."
        ),
    )
    parser.add_argument(
        "--target-spectrum",
        type=Path,
        help=(
            "Path to the HDF file containing the target spectrum. This is "
            "not required of the `model` is an unconditional flow."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name for the output HDF file.",
    )
    args = parser.parse_args()

    return args


def load_model(
    experiment_dir: Path,
    device: torch.device,
) -> tuple[
    Literal["fmpe", "npe", "unconditional_flow"],
    FMPEModel | NPEModel | FlowWrapper,
    dict,
    ThetaScaler,
]:
    """
    Load the model from the given experiment directory.
    """

    # Determine the model type: FMPE / NPE or unconditional flow?
    experiment_config = load_experiment_config(experiment_dir)
    model_type = experiment_config["model"]["model_type"]

    # Type hint for the model
    model: FMPEModel | NPEModel | FlowWrapper

    # Draw samples either from an FMPE / NPE model...
    if model_type in ["fmpe", "npe"]:
        model = build_model(
            experiment_dir=experiment_dir,
            file_path=experiment_dir / "model__best.pt",
            device=device,
        )
        model.network.eval()

    # ... or from an unconditional flow model
    elif model_type == "unconditional_flow":
        checkpoint = torch.load(
            f=experiment_dir / "model__best.pt",
            map_location=device,
        )
        model = create_unconditional_flow_wrapper(
            dim_theta=checkpoint["dim_theta"],
            flow_wrapper_config=experiment_config["model"]["flow_wrapper"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

    else:  # pragma: no cover
        raise ValueError(f"Unknown model type: {model_type}!")

    # Construct the theta scaler
    theta_scaler = get_theta_scaler(config=experiment_config["theta_scaler"])

    # Construct keyword arguments for the model's log_prob method
    if model_type == "fmpe":
        model_kwargs = {"tolerance": 5e-5}
    else:
        model_kwargs = {}

    return model_type, model, model_kwargs, theta_scaler


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print("\nEVALUATE LOG-PROBABILITY UNDER UNCONDITIONAL FLOW\n")

    # Get command line arguments
    args = get_cli_arguments()

    # Define the device
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable device.")
    device = torch.device("cuda")

    # -------------------------------------------------------------------------
    # Load the model under which we want to evaluate the log-probabilities
    # -------------------------------------------------------------------------

    # Define the experiment directory
    experiment_dir = expand_path(args.experiment_dir)
    print(f"Experiment directory: {experiment_dir}\n")

    # Load the model checkpoint
    print("Loading model...", end=" ", flush=True)
    model_type, model, model_kwargs, theta_scaler = load_model(
        experiment_dir=experiment_dir,
        device=device,
    )
    use_amp = model_type == "fmpe"
    print(f"Done!)\n", flush=True)

    # -------------------------------------------------------------------------
    # Load the samples whose log-probabilities we want to evaluate
    # -------------------------------------------------------------------------

    # Load the samples
    print("Loading samples...", end=" ", flush=True)

    file_path = expand_path(args.samples_file)
    with h5py.File(file_path, "r") as hdf_file:
        keys = list(hdf_file.keys())

        # Log-likelihood of samples has a different name, depending on
        # whether the samples came from a ML model or nested sampling
        key = "log_likelihoods" if "log_likelihoods" in keys else "log_l"
        log_likelihoods = np.array(hdf_file[key])[:args.n_samples]

        # Log-prior values: May not always be available
        log_prior_values = np.array(
            hdf_file["log_prior_values"] if "log_prior_values" in keys
            else np.ones(len(log_likelihoods))
        )[:args.n_samples]

        # Samples, weights, and evidence estimate should always be available
        samples = np.array(hdf_file["samples"])[:args.n_samples]
        weights = np.array(hdf_file["weights"])[:args.n_samples]
        log_evidence = np.array(hdf_file["log_evidence"])

    print(f"Done! ({len(samples):,} samples loaded)", flush=True)

    # -------------------------------------------------------------------------
    # Load the target spectrum (if required) and construct context dict
    # -------------------------------------------------------------------------

    context: dict[str, torch.Tensor] | None

    if model_type in ["fmpe", "npe"]:
        target_spectrum = load_target_spectrum(
            file_path=args.target_spectrum,
            index=0,  # hard-code this for now
        )
        context = {
            k: torch.from_numpy(v).float().reshape(1, -1)
            for k, v in target_spectrum.items()
            if k not in ["theta"]
        }
    else:
        context = None

    # -------------------------------------------------------------------------
    # Evaluate log-probabilities under the unconditional flow
    # -------------------------------------------------------------------------

    # Evaluate the log-probability of the samples
    print("Evaluating log-probabilities:", flush=True)

    dataset = TensorDataset(torch.from_numpy(samples).float())
    dataloader = DataLoader(dataset, batch_size=8192)
    log_prob_chunks = []

    for batch in tqdm(dataloader, total=len(dataloader), ncols=80):

        # Scale the samples to the correct range
        theta = theta_scaler.forward_tensor(batch[0]).to(device)

        # FMPE / NPE uses `log_prob_batch`
        if model_type in ["fmpe", "npe"]:
            with torch.no_grad():
                chunk_context = {
                    k: v.repeat(len(theta), 1).to(device, non_blocking=True)
                    for k, v in context.items()
                }
                with torch.cuda.amp.autocast(enabled=use_amp):
                    chunk = model.log_prob_batch(
                        theta=theta,
                        context=chunk_context,
                        **model_kwargs,
                    )
                    log_prob_chunks.append(chunk.cpu().numpy())
                    del chunk

        # Unconditional flow uses `log_prob`
        else:
            with torch.no_grad():
                chunk = model.log_prob(theta=theta)
                log_prob_chunks.append(chunk.cpu().numpy())
                del chunk

    log_probs = np.concatenate(log_prob_chunks, axis=0).flatten()
    print()

    # -------------------------------------------------------------------------
    # Save the results as an HDF file
    # -------------------------------------------------------------------------

    print("Saving results...", end=" ", flush=True)

    # Create the output directory
    output_dir = Path(__file__).parent / "logprobs"
    output_dir.mkdir(exist_ok=True)

    # Save to HDF
    file_path = output_dir / args.output_name
    with h5py.File(file_path, "w") as hdf_file:

        # Store meta-data about input files
        hdf_file.attrs["experiment_dir"] = str(experiment_dir)
        hdf_file.attrs["samples_file"] = str(file_path)
        if args.target_spectrum is not None:
            hdf_file.attrs["target_spectrum"] = str(args.target_spectrum)

        # Store the data that we just copied from the input file
        hdf_file.create_dataset("log_prior_values", data=log_prior_values)
        hdf_file.create_dataset("log_likelihoods", data=log_likelihoods)
        hdf_file.create_dataset("log_evidence", data=log_evidence)
        hdf_file.create_dataset("samples", data=samples)
        hdf_file.create_dataset("weights", data=weights)

        # Store the log-probabilities that we just evaluated
        hdf_file.create_dataset("log_probs", data=log_probs)

    print("Done!\n", flush=True)
    print(f"Output file path:\n{file_path}", flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")
