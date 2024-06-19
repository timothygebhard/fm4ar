"""
Load a trained model and draw some samples (on a GPU).
"""

import argparse
from pathlib import Path
from time import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.models.build_model import FMPEModel, NPEModel, build_model
from fm4ar.target_spectrum import load_target_spectrum

if __name__ == "__main__":

    script_start = time()
    print("\nDRAW SAMPLES\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model__latest.pt",
        help="Name of the checkpoint file. Default: model__latest.pt",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for sampling. Default: 1024.",
    )
    parser.add_argument(
        "--target-file-path",
        type=Path,
        default="$FM4AR_DATASETS_DIR/vasist-2023/test/test__R-400.hdf",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of samples to draw. Default: 10_000.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.125754,
        help="Sigma for the error bars.",
    )
    args = parser.parse_args()

    # Get the device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    # Load the reference spectrum to be used as the context
    print("Loading spectrum...", end=" ", flush=True)
    target_spectrum = load_target_spectrum(
        file_path=args.target_file_path,
        index=args.target_index,
    )
    wlen = target_spectrum["wlen"]
    flux = target_spectrum["flux"]
    print("Done!\n")

    # Print some statistics
    print(f"Mean flux: {flux.mean():.4f}")
    print(f"Std. flux: {flux.std():.4f}\n")

    # Load the model to GPU
    print("Loading model...", end=" ", flush=True)
    model = build_model(
        experiment_dir=args.experiment_dir,
        file_path=args.experiment_dir / args.checkpoint_name,
        device="cuda",
    )
    model.network.eval()
    print("Done!\n")

    # Load the theta scaler
    theta_scaler = get_theta_scaler(model.config["theta_scaler"])

    # Define the target file and make sure it's empty
    file_path = args.experiment_dir / "samples_from_model.hdf"
    with h5py.File(file_path, "w") as f:
        pass

    # Determine the chunk sizes: Every chunk should have `chunk_size` samples,
    # except for the last one, which may have fewer samples.
    chunk_sizes = np.diff(
        np.r_[0 : args.n_samples : args.chunk_size, args.n_samples]
    )

    # Construct the basic context (right error bars, but batch size 1)
    context = {
        "wlen": torch.from_numpy(wlen).float(),
        "flux": torch.from_numpy(flux).float().reshape(1, -1),
        "error_bars": args.sigma * torch.ones(1, len(flux)),
    }

    theta_chunks = []
    print(f"Drawing samples for sigma={args.sigma:.4f}:")
    with torch.no_grad():
        for n in tqdm(chunk_sizes, ncols=80):
            # Adjust the size of the context so that the batch size matches
            # the desired chunk size, and move it to the correct device
            chunk_context = {
                k: v.repeat(n, 1).to("cuda", non_blocking=True)
                for k, v in context.items()
            }

            # Draw samples from the model
            if isinstance(model, NPEModel):
                theta_chunk = model.sample_batch(context=chunk_context)
            elif isinstance(model, FMPEModel):
                theta_chunk = model.sample_batch(
                    context=chunk_context,
                    tolerance=1.0e-3,
                )
            else:
                raise ValueError("Unknown model type!")

            # Inverse-transform the samples
            theta_chunk = theta_scaler.inverse_tensor(theta_chunk.cpu())
            theta_chunks.append(theta_chunk.cpu())

        print(flush=True)

        # Combine all chunks into a single array
        theta = torch.cat(theta_chunks, dim=0).numpy()

        # Save the samples
        with h5py.File(file_path, "a") as f:
            f.create_dataset(
                name="samples",
                data=theta,
                dtype=np.float32,
            )

    # Print the total runtime
    print(f"\nThis took {time() - script_start:.1f} seconds!\n")
