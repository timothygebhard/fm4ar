"""
Measure the inference time of the model.
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from yaml import safe_load

from fm4ar.models.build_model import FMPEModel, build_model
from fm4ar.target_spectrum import load_target_spectrum
from fm4ar.utils.paths import expand_env_variables_in_path as expand_path

if __name__ == "__main__":

    script_start = time()
    print("\nMEASURE INFERENCE TIME\n")

    # Parse command line arguments and load the configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to the configuration file specifying the setup for the "
            "timing experiment (model, target spectrum, n_samples, ...)."
        ),
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = safe_load(f)

    # Get the device (running this script without a GPU does not make sense)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    # Load the reference spectrum to be used as the context
    print("Loading spectrum...", end=" ", flush=True)
    target_spectrum = load_target_spectrum(
        file_path=expand_path(config["target"]["file_path"]),
        index=config["target"]["index"],
    )
    print("Done!")

    # Load the model onto the GPU
    print("Loading model...", end=" ", flush=True)
    model = build_model(
        experiment_dir=None,
        file_path=expand_path(config["model"]["file_path"]),
        device="cuda",
    )
    model.network.eval()
    print("Done!\n\n")

    # Sanity check: number of samples should be a multiple of the chunk size
    # so that we can use the same context dictionary for all chunks
    if config["n_samples"] % config["chunksize"] != 0:
        raise ValueError("n_samples must be a multiple of chunksize!")

    # Construct the basic context (with batch size = 1)
    context = {
        "wlen": torch.from_numpy(target_spectrum["wlen"]).float(),
        "flux": torch.from_numpy(target_spectrum["flux"]).float(),
        "error_bars": torch.from_numpy(target_spectrum["error_bars"]).float(),
    }

    # Repeat the context to match the desired chunk size and move it to the GPU
    chunk_context = {
        k: v.repeat(config["chunksize"], 1).to("cuda", non_blocking=True)
        for k, v in context.items()
    }

    # Use automatic mixed precision for the FMPE model
    use_amp = isinstance(model, FMPEModel)

    # Benchmark the `sample_batch()` method
    times: dict[str, list[float]] = {}
    for label, method, kwargs in [
        (
            "sample",
            model.sample_batch,
            config["model"]["sample_kwargs"]),
        (
            "sample_and_log_prob",
            model.sample_and_log_prob_batch,
            config["model"]["sample_and_logprob_kwargs"]
         ),
    ]:
        print(f"Benchmarking `{label}()`:\n", flush=True)
        times[label] = []
        with torch.no_grad():
            for i in range(config["n_repeats"]):
                start_time = time()
                for _ in range(0, config["n_samples"], config["chunksize"]):
                    with autocast(enabled=use_amp):
                        method(context=chunk_context, **kwargs)
                total_time = time() - start_time
                times[label].append(total_time)
                print(f"[{i:2d}] Total time: {total_time:.2f} s", flush=True)
            mean = np.median(times[label])
            std = np.std(times[label])
            print(f"\nAverage: {mean:.2f} +- {std:.2f}s\n\n", flush=True)

    # Construct data frame and save results to disk
    df = pd.DataFrame(times)
    file_path = "results_" + args.config.stem + ".csv"
    df.to_csv(file_path, index=False)

    # Print the total runtime
    print(f"This took {time() - script_start:.1f} seconds!\n")
